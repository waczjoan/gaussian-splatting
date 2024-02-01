import os
import numpy as np
import trimesh
import torch

from mesh_splatting.utils.graphics_utils import MeshPointCloud
from scene.dataset_readers import (
    readColmapSceneInfo,
    readNerfSyntheticInfo,
    readCamerasFromTransforms,
    getNerfppNorm,
    SceneInfo,
    storePly,
    fetchPly
)
from utils.sh_utils import SH2RGB

from scene.colmap_loader import (
    read_extrinsics_text,
    read_intrinsics_text,
    read_extrinsics_binary,
    read_intrinsics_binary,
    read_points3D_binary,
    read_points3D_text
)

from scene.dataset_readers import readColmapCameras

softmax = torch.nn.Softmax(dim=2)


def readNerfSyntheticMeshInfo(
        path, white_background, eval, num_splats, mesh="mesh", extension=".png"
):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    print("Reading Mesh object")
    mesh_scene = trimesh.load(f'{path}/{mesh}.obj', force='mesh')
    # mesh_scene = trimesh.load(f'{path}/mesh_120.obj', force='mesh')
    vertices = mesh_scene.vertices
    vertices = vertices[:, [0, 2, 1]]
    vertices[:, 1] = -vertices[:, 1]
    # vertices *= 3
    faces = mesh_scene.faces

    triangles = torch.tensor(mesh_scene.triangles).float()  # equal vertices[faces]
    triangles = triangles[:, :, [0, 2, 1]]
    triangles[:, :, 1] = -triangles[:, :, 1]
    # triangles *= 3

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    # if not os.path.exists(ply_path):
    if True:
        # Since this data set has no colmap data, we start with random points
        num_pts_each_triangle = num_splats
        num_pts = num_pts_each_triangle * triangles.shape[0]
        print(
            f"Generating random point cloud ({num_pts})..."
        )

        # We create random points inside the bounds traingles
        alpha = torch.rand(
            triangles.shape[0],
            num_pts_each_triangle,
            3
        )

        xyz = torch.matmul(
            alpha,
            triangles
        )
        xyz = xyz.reshape(num_pts, 3)

        shs = np.random.random((num_pts, 3)) / 255.0

        pcd = MeshPointCloud(
            alpha=alpha,
            points=xyz,
            colors=SH2RGB(shs),
            normals=np.zeros((num_pts, 3)),
            vertices=vertices,
            faces=faces,
            triangles=triangles.cuda()
        )

        storePly(ply_path, pcd.points, SH2RGB(shs) * 255)
    #try:
    #    pcd = fetchPly(ply_path)
    #except:
    #    pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readColmapMeshSceneInfo(path, images, eval, num_splats, meshes, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    if True:
        pcds = []
        ply_paths = []
        total_pts = 0
        for i, (mesh, num) in enumerate(zip(meshes, num_splats)):
            ply_path = os.path.join(path, f"points3d_{i}.ply")

            mesh_scene = trimesh.load(f'{path}/sparse/0/{mesh}.obj', force='mesh')
            vertices = mesh_scene.vertices
            faces = mesh_scene.faces
            triangles = torch.tensor(mesh_scene.triangles).float()  # equal vertices[faces]

            num_pts_each_triangle = num
            num_pts = num_pts_each_triangle * triangles.shape[0]
            total_pts += num_pts

            # We create random points inside the bounds traingles
            alpha = torch.rand(
                triangles.shape[0],
                num_pts_each_triangle,
                3
            )

            xyz = torch.matmul(
                alpha,
                triangles
            )
            xyz = xyz.reshape(num_pts, 3)

            shs = np.random.random((num_pts, 3)) / 255.0

            pcd = MeshPointCloud(
                alpha=alpha,
                points=xyz,
                colors=SH2RGB(shs),
                normals=np.zeros((num_pts, 3)),
                vertices=vertices,
                faces=faces,
                triangles=triangles.cuda()
            )
            pcds.append(pcd)
            ply_paths.append(ply_path)
            storePly(ply_path, pcd.points, SH2RGB(shs) * 255)
    
    print(
        f"Generating random point cloud ({total_pts})..."
    )
    #try:
    #    pcd = fetchPly(ply_path)
    #except:
    #    pcd = None

    scene_info = SceneInfo(point_cloud=pcds,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_paths)
    
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
    "Blender_Mesh": readNerfSyntheticMeshInfo,
    "Colmap_Mesh": readColmapMeshSceneInfo
}
