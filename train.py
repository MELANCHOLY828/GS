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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, edge_aware_normal_loss, ms_ssim, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from tqdm import tqdm
# from utils.image_utils import psnr, render_net_image, depth_to_normal
from utils.image_utils import psnr, render_net_image

from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import random
import math
from utils.vis_utils import apply_depth_colormap, save_points, colormap
from utils.depth_utils import depths_to_points, depth_to_normal
from utils.general_utils import get_expon_lr_func
import torchvision
import numpy as np


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
def depths_double_to_points(view, depthmap1, depthmap2):
    W, H = view.image_width, view.image_height
    fx = W / (2 * math.tan(view.FoVx / 2.))
    fy = H / (2 * math.tan(view.FoVy / 2.))
    intrins = torch.tensor(
        [[fx, 0., W/2.],
        [0., fy, H/2.],
        [0., 0., 1.0]]
    ).float().cuda()
    grid_x, grid_y = torch.meshgrid(torch.arange(W)+0.5, torch.arange(H)+0.5, indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3).float().cuda()
    rays_d = points @ intrins.inverse().T
    # rays_o = torch.zeros(3,dtype=torch.float32,device="cuda")
    # rays_o = c2w[:3,3]
    points1 = depthmap1.reshape(-1, 1) * rays_d
    points2 = depthmap2.reshape(-1, 1) * rays_d
    return points1, points2

def depth_double_to_normal(view, depth1, depth2):
    points1, points2 = depths_double_to_points(view, depth1, depth2)
    points = torch.stack([points1, points2],dim=0).reshape(2, *depth1.shape[1:], 3)
    output = torch.zeros_like(points)
    dx = points[:,2:, 1:-1] - points[:,:-2, 1:-1]
    dy = points[:,1:-1, 2:] - points[:,1:-1, :-2]
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[:,1:-1, 1:-1, :] = normal_map
    return output, points

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    # import pdb
    # pdb.set_trace() 
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    trainCameras = scene.getTrainCameras().copy()
    testCameras = scene.getTestCameras().copy()
    allCameras = trainCameras + testCameras
    
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    img_num = -1
    img_num_modifier = 1
    for iteration in range(first_iter, opt.iterations + 1):
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            if img_num == -1:
                img_num = len(viewpoint_stack)

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1)) 

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg,depth_threshold = opt.depth_threshold * scene.cameras_extent)
        # render_pkg = render(viewpoint_cam, gaussians, pipe, bg)

        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        gs_w = render_pkg["gs_w"]
        rendered_mask: torch.Tensor = render_pkg["mask"]
        rendered_depth: torch.Tensor = render_pkg["depth"]
        rendered_invdepth: torch.Tensor = render_pkg["invdepth"]
        rendered_middepth: torch.Tensor = render_pkg["middepth"]
        rendered_normal: torch.Tensor = render_pkg["normal"]
        depth_distortion: torch.Tensor = render_pkg["depth_distortion"]
        gt_image = viewpoint_cam.original_image.cuda()
        # rendered_depth = rendered_depth / rendered_mask
        # rendered_depth = torch.nan_to_num(rendered_depth, 0, 0)
        # depth = (rendered_depth - rendered_depth.min()) / (rendered_depth.max() - rendered_depth.min())
        depth = rendered_depth/rendered_mask

        if opt.sky or opt.mask_depth:
            mask = viewpoint_cam.mask.cuda().squeeze()
            # gt_image[:,~mask] =0 
            # image[:,~mask] = 0
            if opt.sky:
                gt_image[:,mask] = 0 
            # image[:,mask] = 0
        if iteration >= opt.use_depth_iter and opt.use_depth and viewpoint_cam.original_depth is not None:
            depth_mask = (viewpoint_cam.original_depth>0) # render_pkg["acc"][0]
            gt_maskeddepth = (viewpoint_cam.original_depth * depth_mask).cuda()
            if opt.mask_depth:
                # import pdb
                # pdb.set_trace() 
                gt_maskeddepth[:,mask] = 0
                depth[:,mask] = 0
                gt_maskeddepth[:,~mask] = (gt_maskeddepth[:,~mask] - gt_maskeddepth[:,~mask].min()) / (gt_maskeddepth[:,~mask].max() - gt_maskeddepth[:,~mask].min())
                depth[:,~mask] = (depth[:,~mask] - depth[:,~mask].min()) / (depth[:,~mask].max() - depth[:,~mask].min())

        # # 转换Tensor为numpy数组，并将其值从[0,1]范围缩放到[0,255]范围
        # rgb_array = (gt_image * 255).byte().cpu().numpy()

        # # 变换数组的形状从 [3, 545, 980] 到 [545, 980, 3]
        # rgb_array = np.transpose(rgb_array, (1, 2, 0))

        # # 将numpy数组转换为PIL图片
        # image = Image.fromarray(rgb_array)

        # # 保存图片
        # image.save('output_image_pil.png')
        if iteration >= opt.use_depth_iter and opt.use_depth and viewpoint_cam.original_depth is not None:
            deploss = l1_loss(gt_maskeddepth, rendered_invdepth*depth_mask) * depth_l1_weight(iteration)
        else:
            deploss = torch.tensor([0],dtype=torch.float32,device="cuda")

        edge = viewpoint_cam.edge
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        L_ms_ssim = (1.0 - ms_ssim(image, gt_image))
        Lrgb = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        # Lrgb =  (1.0 - opt.lambda_ms_ssim) * Ll1 + opt.lambda_ms_ssim * L_ms_ssim 
        # loss = Lrgb
        
        # if iteration < opt.atom_proliferation_until:
        #     Lnormal = edge_aware_normal_loss(gt_image, depth_to_normal(render_pkg["mean_depth"], viewpoint_cam).permute(2,0,1))
        #     loss += opt.lambda_normal*Lnormal

        if iteration >= opt.regularization_from_iter:
            # depth distortion loss
            lambda_distortion = opt.lambda_distortion
            depth_distortion = torch.where(rendered_mask>0,depth_distortion/(rendered_mask * rendered_mask).detach(),0)
            distortion_map = depth_distortion[0] * edge
            distortion_loss = distortion_map.mean()

            # normal consistency loss
            rendered_depth = rendered_depth / rendered_mask
            rendered_depth = torch.nan_to_num(rendered_depth, 0, 0)
            depth_middepth_normal, _ = depth_double_to_normal(viewpoint_cam, rendered_depth, rendered_middepth)
            depth_ratio = 0.6
            rendered_normal = torch.nn.functional.normalize(rendered_normal, p=2, dim=0)
            rendered_normal = rendered_normal.permute(1,2,0)
            normal_error_map = (1 - (rendered_normal.unsqueeze(0) * depth_middepth_normal).sum(dim=-1))
            depth_normal_loss = (1-depth_ratio) * normal_error_map[0].mean() + depth_ratio * normal_error_map[1].mean()
            lambda_depth_normal = opt.lambda_depth_normal
        else:
            lambda_distortion = 0
            lambda_depth_normal = 0
            distortion_loss = torch.tensor([0],dtype=torch.float32,device="cuda")
            depth_normal_loss = torch.tensor([0],dtype=torch.float32,device="cuda")
            
        
        loss = Lrgb + depth_normal_loss * lambda_depth_normal + distortion_loss * lambda_distortion + deploss
        
        # scale_loss = True
        # lambda_ = 1e-5
        
        if opt.scale_loss and iteration>=7000:
            scales = gaussians.get_scaling   
            s1, s2, s3 = scales[:, 0], scales[:, 1], scales[:, 2]
            sum_of_squares = s1**2 + s2**2 + s3**2
            s1_normalized = (s1**2) / (sum_of_squares)
            s2_normalized = (s2**2) / (sum_of_squares)
            s3_normalized = (s3**2) / (sum_of_squares)
            # 计算 h = -累加和(si_normalized * log(si_normalized))，并分别计算三个维度
            h1 = (s1_normalized * torch.log(s1_normalized))
            h2 = (s2_normalized * torch.log(s2_normalized))
            h3 = (s3_normalized * torch.log(s3_normalized))

            # 将三个维度的 h 相加
            h = -(h1 + h2 + h3)
            erank = torch.exp(h)
            log_h = -torch.log(erank-1+opt.lambda_rank)
            loss_rank = opt.lambda_erank * torch.max(log_h, torch.tensor(0.0))
            loss_rank = loss_rank.mean()
            # loss_rank = torch.sum(loss_rank)
            min_scale, _ = torch.min(scales, dim=1)
            min_scale = torch.clamp(min_scale, 0, 30)
            scale_loss = torch.abs(min_scale).mean() + loss_rank
            # scale_loss = torch.sum(torch.abs(min_scale)) + loss_rank
            loss += scale_loss


        loss.backward()

        iter_end.record()
        is_save_images = True
        if is_save_images and (iteration % opt.densification_interval == 0):
            log_depth = rendered_invdepth.expand(3, rendered_invdepth.shape[1], rendered_invdepth.shape[2])
            if iteration >= opt.use_depth_iter and opt.use_depth and viewpoint_cam.original_depth is not None:
                gt_maskeddepth = gt_maskeddepth.expand(3, gt_maskeddepth.shape[1], gt_maskeddepth.shape[2])
                row0 = torch.cat([gt_image, image, gt_maskeddepth, log_depth], dim=2)       
            else:
                # original_depth = viewpoint_cam.original_depth.expand(3, viewpoint_cam.original_depth.shape[1], viewpoint_cam.original_depth.shape[2])

                row0 = torch.cat([gt_image, image], dim=2)       
            image_to_show = torch.clamp(row0, 0, 1)
            
            os.makedirs(f"{dataset.model_path}/log_images", exist_ok = True)
            torchvision.utils.save_image(image_to_show, f"{dataset.model_path}/log_images/{iteration}.jpg")
        

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "#": f"{gaussians.get_opacity.shape[0]}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            gaussians.max_weight[visibility_filter] = torch.max(gaussians.max_weight[visibility_filter],
                                                                gs_w[visibility_filter])
            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])

                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None   #改成3000了
                    # gaussians.densify_and_prune(opt.densify_grad_threshold, 0.05, scene.cameras_extent, size_threshold)
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.densify_grad_abs_threshold, 0.005, scene.cameras_extent, size_threshold)

                    # gaussians.densify_and_prune(opt.clone_threshold, min(opt.split_threshold*iteration/opt.warm_up_until, opt.split_threshold), opt.prune_threshold)

                if  iteration % opt.densification_interval == 0 and iteration < opt.atom_proliferation_until:
                    gaussians.atomize()

                if iteration % opt.opacity_reduce_interval == 0 and opt.use_reduce:
                    gaussians.reduce_opacity()

                if (iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter)):
                    gaussians.reset_opacity()
            if iteration > opt.densify_from_iter and iteration < opt.prune_until_iter and opt.use_prune_weight:
                if iteration % img_num / img_num_modifier == 0 and iteration % opt.opacity_reset_interval > img_num / img_num_modifier:
                    prune_mask = (gaussians.max_weight < opt.min_weight).squeeze()
                    gaussians.prune_points(prune_mask)
                    gaussians.max_weight *= 0
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                
        with torch.no_grad():        
            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)   
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log
                        # Add more metrics as needed
                    }
                    # Send the data
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    # raise e
                    network_gui.conn = None

def prepare_output_and_logger(args):    
    if not args.model_path:
        args.model_path = os.path.join("./output/", os.path.basename(args.source_path))
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    # parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    # parser.add_argument("--save_iterations", nargs="+", type=int, default=[1_000,2_000,3_000,4_000,5_000,6_000,7_000, 8_000,9_000,10_000,30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
