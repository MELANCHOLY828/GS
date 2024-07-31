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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
import cv2
class Camera(nn.Module):
    def __init__(self, resolution, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask, mask, depth,depth_params,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        if mask != None:
            self.mask = mask.bool().to(self.data_device)
        else :
            self.mask = None
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
        self.original_depth = None
        if depth is not None and depth_params is not None and depth_params["scale"] > 0:
            depthmapScaled = depth * depth_params["scale"] + depth_params["offset"]
            depthmapScaled = cv2.resize(depthmapScaled, resolution)
            depthmapScaled[depthmapScaled < 0] = 0
            if depthmapScaled.ndim != 2:
                depthmapScaled = depthmapScaled[..., 0]
            self.original_depth = torch.from_numpy(depthmapScaled[None]).to(self.data_device)
            
            # if self.alpha_mask is not None:
            #     self.depth_mask = self.alpha_mask.clone()
            # else:
            #     self.depth_mask = torch.ones_like(self.original_depth > 0)
            
            # if depth_params["scale"] < 0.2 * depth_params["med_scale"] or depth_params["scale"] > 5 * depth_params["med_scale"]: 
            #     self.depth_mask *= 0
            # else:
            #     self.depth_reliable = True
        # self.original_depth = depth.to(self.data_device) if depth is not None else None

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
            if depth is not None:
                self.original_depth *= gt_alpha_mask[0].to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        
        with torch.no_grad():
            grad_img_left = torch.mean(torch.abs(self.original_image[:, 1:-1, 1:-1] - self.original_image[:, 1:-1, :-2]), 0)
            grad_img_right = torch.mean(torch.abs(self.original_image[:, 1:-1, 1:-1] - self.original_image[:, 1:-1, 2:]), 0)
            grad_img_top = torch.mean(torch.abs(self.original_image[:, 1:-1, 1:-1] - self.original_image[:, :-2, 1:-1]), 0)
            grad_img_bottom = torch.mean(torch.abs(self.original_image[:, 1:-1, 1:-1] - self.original_image[:, 2:, 1:-1]), 0)
            max_grad = torch.max(torch.stack([grad_img_left, grad_img_right, grad_img_top, grad_img_bottom], dim=-1), dim=-1)[0]
            # pad
            max_grad = torch.exp(-max_grad)
            max_grad = torch.nn.functional.pad(max_grad, (1, 1, 1, 1), mode="constant", value=0)
            self.edge = max_grad
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
        self.projection_matrix = torch.bmm(self.world_view_transform.unsqueeze(0).inverse(), self.full_proj_transform.unsqueeze(0)).squeeze(0)


