import torch
import numpy as np
import os

from torch import nn

from scene.gaussian_model import GaussianModel
from simple_knn._C import distCUDA2
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from utils.sh_utils import RGB2SH
from mesh_splatting.utils.graphics_utils import MeshPointCloud
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.system_utils import mkdir_p


class GaussianFlameModel(GaussianModel):

    def __init__(self, sh_degree: int):

        super().__init__(sh_degree)
        self.point_claud = None
        self._alpha = torch.empty(0)
        self.alpha = torch.empty(0)
        self.softmax = torch.nn.Softmax(dim=2)

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.update_alpha_func = self.softmax

        self.vertices = None
        self.faces = None


    @property
    def get_xyz(self):
        return self._xyz

    def create_from_pcd(self, pcd: MeshPointCloud, spatial_lr_scale: float):

        self.point_claud = pcd
        self.spatial_lr_scale = spatial_lr_scale
        pcd_alpha_shape = pcd.alpha.shape

        print("Number of faces: ", pcd_alpha_shape[0])
        print("Number of points at initialisation in face: ", pcd_alpha_shape[1])

        alpha_point_cloud = pcd.alpha.float().cuda()

        print("Number of points at initialisation : ",
              alpha_point_cloud.shape[0] * alpha_point_cloud.shape[1])

        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        # TODO dist2, scales, rots, opacities
        dist2 = torch.clamp_min(
            distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001
        )

        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((pcd.points.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((pcd.points.shape[0], 1), dtype=torch.float, device="cuda"))

        self.create_flame_params()

        self._alpha = nn.Parameter(alpha_point_cloud.requires_grad_(True))  # check update_alpha
        self.update_alpha()
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def create_flame_params(self):
        """
        Create manipulation parameters FLAME model.

        Each parameter is responsible for something different,
        respectively: shape, facial expression, etc.
        """
        self._flame_shape = nn.Parameter(self.point_claud.flame_model_shape_init.requires_grad_(True))
        self._flame_exp = nn.Parameter(self.point_claud.flame_model_expression_init.requires_grad_(True))
        self._flame_pose = nn.Parameter(self.point_claud.flame_model_pose_init.requires_grad_(True))
        self._flame_neck_pose = nn.Parameter(self.point_claud.flame_model_neck_pose_init.requires_grad_(True))
        self._flame_trans = nn.Parameter(self.point_claud.flame_model_transl_init.requires_grad_(True))
        self.faces = self.point_claud.faces

        vertices_enlargement = torch.ones_like(self.point_claud.vertices_init).requires_grad_(True)
        self._vertices_enlargement = nn.Parameter(self.point_claud.vertices_enlargement_init * vertices_enlargement)

    def _calc_xyz(self):
        """
        calculate the 3d Gaussian center in the coordinates xyz.

        The alphas that are taken into account are the distances
        to the vertices and the coordinates of
        the triangles forming the mesh.

        """
        _xyz = torch.matmul(
            self.alpha,
            self.vertices[self.faces]
        )
        self._xyz = _xyz.reshape(
                _xyz.shape[0] * _xyz.shape[1], 3
            )

    def update_alpha(self):
        """
        Function to control the alpha value.

        Alpha is the distance of the center of the gauss
         from the vertex of the triangle of the mesh.
        Thus, for each center of the gauss, 3 alphas
        are determined: alpha1+ alpha2+ alpha3.
        For a point to be in the center of the vertex,
        the alphas must meet the assumptions:
        alpha1 + alpha2 + alpha3 = 1
        and alpha1 + alpha2 +alpha3 >= 0

        #TODO
        check:
        # self.alpha = torch.relu(self._alpha)
        # self.alpha = self.alpha / self.alpha.sum(dim=-1, keepdim=True)

        """
        self.alpha = self.update_alpha_func(self._alpha)
        vertices, _ = self.point_claud.flame_model(
            shape_params=self._flame_shape,
            expression_params=self._flame_exp,
            pose_params=self._flame_pose,
            neck_pose=self._flame_neck_pose,
            transl=self._flame_trans
        )
        self.vertices = self.point_claud.transform_vertices_function(
            vertices,
            self._vertices_enlargement
        )
        self._calc_xyz()

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        lr = 0.00016

        l = [
            {'params': [self._flame_shape], 'lr': lr, "name": "shape"},
            {'params': [self._flame_exp], 'lr': lr, "name": "expression"},
            {'params': [self._flame_pose], 'lr': lr, "name": "pose"},
            {'params': [self._flame_neck_pose], 'lr': lr, "name": "neck_pose"},
            {'params': [self._flame_trans], 'lr': lr, "name": "transl"},
            {'params': [self._vertices_enlargement], 'lr': 0.0001, "name": "vertices_enlargement"},
            {'params': [self._alpha], 'lr': 0.001, "name": "alpha"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.flame_scheduler_args = get_expon_lr_func(
            lr_init=lr,
            lr_final=0.0001,
            max_steps=training_args.iterations
        )
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        list_params = ['shape', 'expression', 'pose',
                       'neck_pose', 'transl']
        for param_group in self.optimizer.param_groups:
            if param_group["name"] in list_params:
                lr = self.flame_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr


    def save_ply(self, path):
        self._save_ply(path)

        attrs = self.__dict__
        flame_additional_attrs = [
            '_flame_shape', '_flame_exp', '_flame_pose',
            '_flame_neck_pose',
            '_flame_trans',
            '_vertices_enlargement', 'faces',
            'alpha', 'point_claud',

        ]

        save_dict = {}
        for attr_name in flame_additional_attrs:
            save_dict[attr_name] = attrs[attr_name]

        path_flame = path.replace('point_cloud.ply', 'flame_params.pt')
        torch.save(save_dict, path_flame)

    def load_ply(self, path):
        self._load_ply(path)
        path_flame = path.replace('point_cloud.ply', 'flame_params.pt')
        params = torch.load(path_flame)
        self._flame_shape = params['_flame_shape']
        self._flame_exp = params['_flame_exp']
        self._flame_pose = params['_flame_pose']
        self._flame_neck_pose = params['_flame_neck_pose']
        self._flame_trans = params['_flame_trans']
        self._vertices_enlargement = params['_vertices_enlargement']
        self.faces = params['faces']
        self.alpha = params['alpha']
        self.point_claud = params['point_claud']