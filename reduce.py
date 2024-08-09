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
from gaussian_renderer import GaussianModel
import pandas as pd
import numpy as np
models_configuration = {
    'baseline': {
        'quantised': False,
        'half_float': False,
        'name': 'point_cloud.ply'
        },
    'quantised': {
        'quantised': True,
        'half_float': False,
        'name': 'point_cloud_quantised.ply'
        },
    'quantised_half': {
        'quantised': True,
        'half_float': True,
        'name': 'point_cloud_quantised_half.ply'
        },
}

def measure_fps(iteration, views, gaussians, pipeline, background, pcd_name):
    fps = 0
    for _, view in enumerate(views):
        render(view, gaussians, pipeline, background, measure_fps=False)
    for _, view in enumerate(views):
        fps += render(view, gaussians, pipeline, background, measure_fps=True)["FPS"]

    fps *= 1000 / len(views)
    return pd.Series([fps], index=["FPS"], name=f"{pcd_name}_{iteration}")


def render_set(model_path,
               name,
               iteration,
               views,
               gaussians,
               pipeline,
               background,
               pcd_name):
    render_path = os.path.join(model_path, name, f"{pcd_name}_{iteration}", "renders")
    gts_path = os.path.join(model_path, name, f"{pcd_name}_{iteration}", "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def reduce_sets(dataset : ModelParams,
                iteration : int,
                pipeline : PipelineParams,
                skip_train : bool,
                skip_test : bool,
                skip_measure_fps : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)


        gaussians.cull_sh_bands(scene.getTrainCameras(), threshold=args.cdist_threshold*np.sqrt(3)/255, std_threshold=args.std_threshold)
        scene.save_reduce(iteration)
        scene.gaussians.produce_clusters(store_dict_path=scene.model_path)  
        scene.save_reduce(iteration, quantise=True)
        scene.save_reduce(iteration, quantise=True, half_float=True)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--models",
                        help="Types of models to test",
                        choices=models_configuration.keys(),
                        default=['baseline', 'quantised_half'],
                        nargs="+")
    parser.add_argument("--skip_measure_fps", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--cdist_threshold", default=6, type=int)
    parser.add_argument("--std_threshold", default=0.04, type=float)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    reduce_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_measure_fps)
