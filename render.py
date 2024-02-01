#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from mesh_splatting.scene.gaussian_mesh_model import GaussianMeshModel
from multi_mesh_splatting.scene.gaussian_multi_mesh_model import GaussianMultiMeshModel

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = []
        gaussians1 = GaussianMultiMeshModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians1, load_iteration=iteration, shuffle=False)
        if hasattr(gaussians1, 'update_alpha'):
            gaussians1.update_alpha()
        if hasattr(gaussians1, 'prepare_scaling_rot'):
            gaussians1.prepare_scaling_rot()
        gaussians.append(gaussians1)

        # gaussians2 = GaussianMeshModel(dataset.sh_degree)
        # dataset.model_path = "output/ficus/10_w"
        # dataset.source_path = "/home/pieczo/forks/gaussian-splatting/data/ficus"
        # dataset.white_background = True
        # dataset.num_splats = [10]
        # dataset.meshes = ["mesh"]
        # _ = Scene(dataset, gaussians2, load_iteration=iteration, shuffle=False)
        # dataset.white_background = False
        # dataset.model_path = scene.model_path
        # gaussians.append(gaussians2)

        # gaussians3 = GaussianMeshModel(dataset.sh_degree)
        # dataset.model_path = "output/ficus/20"
        # dataset.source_path = "/home/pieczo/forks/gaussian-splatting/data/ficus"
        # dataset.white_background = True
        # dataset.num_splats = [20]
        # dataset.meshes = ["teaser2"]
        # _ = Scene(dataset, gaussians3, load_iteration=iteration, shuffle=False)
        # dataset.white_background = False
        # dataset.model_path = scene.model_path
        # gaussians.append(gaussians3)

        # gaussians4 = GaussianMeshModel(dataset.sh_degree)
        # dataset.model_path = "output/ficus/20"
        # dataset.source_path = "/home/pieczo/forks/gaussian-splatting/data/ficus"
        # dataset.white_background = True
        # dataset.num_splats = [20]
        # dataset.meshes = ["teaser2"]
        # _ = Scene(dataset, gaussians4, load_iteration=iteration, shuffle=False)
        # dataset.white_background = False
        # dataset.model_path = scene.model_path
        # gaussians.append(gaussians4)

        # gaussians5 = GaussianMeshModel(dataset.sh_degree)
        # dataset.model_path = "output/ficus/20"
        # dataset.source_path = "/home/pieczo/forks/gaussian-splatting/data/ficus"
        # dataset.white_background = True
        # dataset.num_splats = [20]
        # dataset.meshes = ["teaser3"]
        # _ = Scene(dataset, gaussians5, load_iteration=iteration, shuffle=False)
        # dataset.white_background = False
        # dataset.model_path = scene.model_path
        # gaussians.append(gaussians5)
        
        # gaussians6 = GaussianMeshModel(dataset.sh_degree)
        # dataset.model_path = "output/ficus/20"
        # dataset.source_path = "/home/pieczo/forks/gaussian-splatting/data/ficus"
        # dataset.white_background = True
        # dataset.num_splats = [20]
        # dataset.meshes = ["teaser4"]
        # _ = Scene(dataset, gaussians6, load_iteration=iteration, shuffle=False)
        # dataset.white_background = False
        # dataset.model_path = scene.model_path
        # gaussians.append(gaussians6)     

        gaussians7 = GaussianMeshModel(dataset.sh_degree)
        dataset.model_path = "output/hotdog/animate_white"
        dataset.source_path = "/home/pieczo/forks/gaussian-splatting/data/hotdog"
        dataset.white_background = True
        dataset.num_splats = [20]
        dataset.meshes = ["teaser5"]
        _ = Scene(dataset, gaussians7, load_iteration=iteration, shuffle=False)
        dataset.white_background = False
        dataset.model_path = scene.model_path
        gaussians.append(gaussians7)

        gaussians8 = GaussianMeshModel(dataset.sh_degree)
        dataset.model_path = "output/lego/white_animate"
        dataset.source_path = "/home/pieczo/forks/gaussian-splatting/data/lego"
        dataset.white_background = True
        dataset.num_splats = [1]
        dataset.meshes = ["teaser6"]
        _ = Scene(dataset, gaussians8, load_iteration=iteration, shuffle=False)
        dataset.white_background = False
        dataset.model_path = scene.model_path
        gaussians.append(gaussians8)

        gaussians9 = GaussianMeshModel(dataset.sh_degree)
        dataset.model_path = "output/lego/white_animate"
        dataset.source_path = "/home/pieczo/forks/gaussian-splatting/data/lego"
        dataset.white_background = True
        dataset.num_splats = [1]
        dataset.meshes = ["teaser7"]
        _ = Scene(dataset, gaussians9, load_iteration=iteration, shuffle=False)
        dataset.white_background = False
        dataset.model_path = scene.model_path
        gaussians.append(gaussians9)
        for g in gaussians:
            g.update_alpha()
            g.prepare_scaling_rot()
        # gaussians = [gaussians1, gaussians2, gaussians3, gaussians4, gaussians5, gaussians6, gaussians7, gaussians8]

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "composition", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--num_splats", nargs="+", type=int, default=[])
    parser.add_argument("--meshes", nargs="+", type=str, default=[])
    args = get_combined_args(parser)
    model.num_splats = args.num_splats
    model.meshes = args.meshes
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)