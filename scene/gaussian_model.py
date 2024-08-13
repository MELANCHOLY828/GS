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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH, SH2RGB
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from diff_gaussian_rasterization._C import calculate_colours_variance, kmeans_cuda
from simple_knn._C import distCUDA2
from collections import OrderedDict

def pcast_i16_to_f32(tensor):
    return tensor.view(dtype=torch.float16).float()
class Codebook():
    def __init__(self, ids, centers):
        self.ids = ids
        self.centers = centers
    
    def evaluate(self):
        return self.centers[self.ids.flatten().long()]
def generate_codebook(values, inverse_activation_fn=lambda x: x, num_clusters=256, tol=0.0001):
    shape = values.shape
    values = values.flatten().view(-1, 1)
    centers = values[torch.randint(values.shape[0], (num_clusters, 1), device="cuda").squeeze()].view(-1,1)

    ids, centers = kmeans_cuda(values, centers.squeeze(), tol, 500)
    ids = ids.byte().squeeze().view(shape)
    centers = centers.view(-1,1)

    return Codebook(ids.cuda(), inverse_activation_fn(centers.cuda()))
class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.spatial_lr_scale = 0
        self.max_radii2D = torch.empty(0)
        self.percent_dense = 0
        self.max_weight = torch.empty(0)
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.max_weight,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.max_radii2D,
        self.max_weight,
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, atom_init_quantile : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist = torch.sqrt(torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001))
        self.atom_scale = torch.quantile(dist, torch.tensor([atom_init_quantile]).cuda()).item()
        dist[dist<self.atom_scale] = self.atom_scale
        # self.atom_scale = 0.01 * self.spatial_lr_scale
        # dist = torch.ones((fused_point_cloud.shape[0]), device="cuda") * self.atom_scale
        scales = torch.log(dist)[...,None].repeat(1, 3)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.max_weight = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs_max = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self, rest_coeffs = None):   #不一样
        if rest_coeffs == None:
            l = ['x', 'y', 'z', 'red', 'green', 'blue']
        else:
            l = ['x', 'y', 'z']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        if rest_coeffs is None:
            rest_coeffs = self._features_rest.shape[1]*self._features_rest.shape[2]
        for i in range(rest_coeffs):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l
    
    # def construct_list_of_attributes(self, rest_coeffs = 45):   #不一样
    #     l = ['x', 'y', 'z', 'red', 'green', 'blue']
    #     # All channels except the 3 DC
    #     for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
    #         l.append('f_dc_{}'.format(i))
    #     for i in range(rest_coeffs):
    #         l.append('f_rest_{}'.format(i))
    #     l.append('opacity')
    #     for i in range(self._scaling.shape[1]):
    #         l.append('scale_{}'.format(i))
    #     for i in range(self._rotation.shape[1]):
    #         l.append('rot_{}'.format(i))
    #     return l
    def produce_clusters(self, num_clusters=256, store_dict_path=None):
        max_coeffs_num = (self.max_sh_degree + 1)**2 - 1
        codebook_dict = OrderedDict({})

        codebook_dict["features_dc"] = generate_codebook(self._features_dc.detach()[:, 0],
                                                         num_clusters=num_clusters, tol=0.001)
        for sh_degree in range(max_coeffs_num):
                codebook_dict[f"features_rest_{sh_degree}"] = generate_codebook(
                    self._features_rest.detach()[:, sh_degree], num_clusters=num_clusters)

        codebook_dict["opacity"] = generate_codebook(self.get_opacity.detach(),
                                                     self.inverse_opacity_activation, num_clusters=num_clusters)
        codebook_dict["scaling"] = generate_codebook(self.get_scaling.detach(),
                                                     self.scaling_inverse_activation, num_clusters=num_clusters)
        codebook_dict["rotation_re"] = generate_codebook(self.get_rotation.detach()[:, 0:1],
                                                         num_clusters=num_clusters)
        codebook_dict["rotation_im"] = generate_codebook(self.get_rotation.detach()[:, 1:],
                                                         num_clusters=num_clusters)
        if store_dict_path is not None:
            torch.save(codebook_dict, os.path.join(store_dict_path, 'codebook.pt'))
        
        self._codebook_dict = codebook_dict
    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))
        xyz = self._xyz.detach().cpu().numpy()
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        rgb = np.clip(SH2RGB(f_dc), 0, 1) * 255
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, rgb, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def save_ply_reduce(self, path, quantised=False, half_float=False, variable_sh_bands=False):
        self.variable_sh_bands = variable_sh_bands
        float_type = 'int16' if half_float else 'f4'
        attribute_type = 'u1' if quantised else float_type
        max_sh_coeffs = (self.max_sh_degree + 1) ** 2 - 1
        mkdir_p(os.path.dirname(path))
        elements_list = []

        if quantised:
            # Read codebook dict to extract ids and centers
            if self._codebook_dict is None:
                print("Clustering codebook missing. Returning without saving")
                return

            opacity = self._codebook_dict["opacity"].ids
            scaling = self._codebook_dict["scaling"].ids
            rot = torch.cat((self._codebook_dict["rotation_re"].ids,
                            self._codebook_dict["rotation_im"].ids),
                            dim=1)
            features_dc = self._codebook_dict["features_dc"].ids
            features_rest = torch.stack([self._codebook_dict[f"features_rest_{i}"].ids
                                        for i in range(max_sh_coeffs)
                                        ], dim=1).squeeze()

            dtype_full = [(k, float_type) for k in self._codebook_dict.keys()]
            codebooks = np.empty(256, dtype=dtype_full)

            centers_numpy_list = [v.centers.detach().cpu().numpy() for v in self._codebook_dict.values()]

            if half_float:
                # No float 16 for plydata, so we just pointer cast everything to int16
                for i in range(len(centers_numpy_list)):
                    centers_numpy_list[i] = np.cast[np.float16](centers_numpy_list[i]).view(dtype=np.int16)
                
            codebooks[:] = list(map(tuple, np.concatenate([ar for ar in centers_numpy_list], axis=1)))
                
        else:
            opacity = self._opacity
            scaling = self._scaling
            rot = self._rotation
            features_dc = self._features_dc
            features_rest = self._features_rest

        for sh_degree in range(self.max_sh_degree + 1):
            coeffs_num = (sh_degree+1)**2 - 1
            degrees_mask = (self._degrees == sh_degree).squeeze()

            #  Position is not quantised
            if half_float:
                xyz = self._xyz[degrees_mask].detach().cpu().half().view(dtype=torch.int16).numpy()
            else:
                xyz = self._xyz[degrees_mask].detach().cpu().numpy()

            f_dc = features_dc[degrees_mask].detach().contiguous().cpu().view(-1,3).numpy()
            # Transpose so that to save rest featrues as rrr ggg bbb instead of rgb rgb rgb
            if self.variable_sh_bands:
                f_rest = features_rest[sh_degree][:, :coeffs_num].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            else:
                f_rest = features_rest[degrees_mask][:, :coeffs_num].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            opacities = opacity[degrees_mask].detach().cpu().numpy()
            scale = scaling[degrees_mask].detach().cpu().numpy()
            rotation = rot[degrees_mask].detach().cpu().numpy()

            dtype_full = [(attribute, float_type) 
                          if attribute in ['x', 'y', 'z'] else (attribute, attribute_type) 
                          for attribute in self.construct_list_of_attributes(coeffs_num * 3)]
            elements = np.empty(degrees_mask.sum(), dtype=dtype_full)

            attributes = np.concatenate((xyz, f_dc, f_rest, opacities, scale, rotation), axis=1)
            elements[:] = list(map(tuple, attributes))
            elements_list.append(PlyElement.describe(elements, f'vertex_{sh_degree}'))
        if quantised:
            elements_list.append(PlyElement.describe(codebooks, f'codebook_centers'))
        PlyData(elements_list).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def reduce_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.8))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def atomize(self):     #加的 
        dist = self.get_scaling
        atom_mask = torch.min(self.get_scaling, dim=1).values <= self.atom_scale
        dist[atom_mask] = self.atom_scale
        
        scaling_new = torch.log(dist)
        optimizable_tensors = self.replace_tensor_to_optimizer(scaling_new, "scaling")
        self._scaling = optimizable_tensors["scaling"]
        # geometric progression ensures the atom scale will be optimized into 1/2 at 7k iteration
        self.atom_scale*=0.99
        dist[atom_mask]*=0.99
    def _parse_vertex_group(self,
                            vertex_group,
                            sh_degree,
                            float_type,
                            attribute_type,
                            max_coeffs_num,
                            quantised,
                            half_precision,                            
                            codebook_centers_torch=None):
        coeffs_num = (sh_degree+1)**2 - 1
        num_primitives = vertex_group.count

        xyz = np.stack((np.asarray(vertex_group["x"], dtype=float_type),
                        np.asarray(vertex_group["y"], dtype=float_type),
                        np.asarray(vertex_group["z"], dtype=float_type)), axis=1)

        opacity = np.asarray(vertex_group["opacity"], dtype=attribute_type)[..., np.newaxis]
    
        # Stacks the separate components of a vector attribute into a joint numpy array
        # Defined just to avoid visual clutter
        def stack_vector_attribute(name, count):
            return np.stack([np.asarray(vertex_group[f"{name}_{i}"], dtype=attribute_type)
                            for i in range(count)], axis=1)

        features_dc = stack_vector_attribute("f_dc", 3).reshape(-1, 1, 3)
        scaling = stack_vector_attribute("scale", 3)
        rotation = stack_vector_attribute("rot", 4)
        
        # Take care of error when trying to stack 0 arrays
        if sh_degree > 0:
            features_rest = stack_vector_attribute("f_rest", coeffs_num*3).reshape((num_primitives, 3, coeffs_num))
        else:
            features_rest = np.empty((num_primitives, 3, 0), dtype=attribute_type)

        if not self.variable_sh_bands:
            # Using full tensors (P x 15) even for points that don't require it
            features_rest = np.concatenate(
                (features_rest,
                    np.zeros((num_primitives, 3, max_coeffs_num - coeffs_num), dtype=attribute_type)), axis=2)

        degrees = np.ones(num_primitives, dtype=np.int32)[..., np.newaxis] * sh_degree

        xyz = torch.from_numpy(xyz).cuda()
        if half_precision:
            xyz = pcast_i16_to_f32(xyz)
        features_dc = torch.from_numpy(features_dc).contiguous().cuda()
        features_rest = torch.from_numpy(features_rest).contiguous().cuda()
        opacity = torch.from_numpy(opacity).cuda()
        scaling = torch.from_numpy(scaling).cuda()
        rotation = torch.from_numpy(rotation).cuda()
        degrees = torch.from_numpy(degrees).cuda()

        # If quantisation has been used, it is needed to index the centers
        if quantised:
            features_dc = codebook_centers_torch['features_dc'][features_dc.view(-1).long()].view(-1, 1, 3)

            # This is needed as we might have padded the features_rest tensor with zeros before
            reshape_channels = coeffs_num if self.variable_sh_bands else max_coeffs_num            
            # The gather operation indexes a 256x15 tensor with a (P*3)features_rest index tensor,
            # in a column-wise fashion
            # Basically this is equivalent to indexing a single codebook with a P*3 index
            # features_rest times inside a loop
            features_rest = codebook_centers_torch['features_rest'].gather(0, features_rest.view(num_primitives*3, reshape_channels).long()).view(num_primitives, 3, reshape_channels)
            opacity = codebook_centers_torch['opacity'][opacity.long()]
            scaling = codebook_centers_torch['scaling'][scaling.view(num_primitives*3).long()].view(num_primitives, 3)
            # Index the real and imaginary part separately
            rotation = torch.cat((
                codebook_centers_torch['rotation_re'][rotation[:, 0:1].long()],
                codebook_centers_torch['rotation_im'][rotation[:, 1:].reshape(num_primitives*3).long()].view(num_primitives,3)
                ), dim=1)

        return {'xyz': xyz,
                'opacity': opacity,
                'features_dc': features_dc,
                'features_rest': features_rest,
                'scaling': scaling,
                'rotation': rotation,
                'degrees': degrees
        }
    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def load_ply_reduce(self, path, half_float=False, quantised=False, variable_sh_bands = False):
        self.variable_sh_bands = variable_sh_bands
        plydata = PlyData.read(path)

        xyz_list = []
        features_dc_list = []
        features_rest_list = []
        opacity_list = []
        scaling_list = []
        rotation_list = []
        degrees_list = []

        float_type = 'int16' if half_float else 'f4'
        attribute_type = 'u1' if quantised else float_type
        max_coeffs_num = (self.max_sh_degree+1)**2 - 1

        codebook_centers_torch = None
        if quantised:
            # Parse the codebooks.
            # The layout is 256 x 20, where 256 is the number of centers and 20 number of codebooks
            # In the future we could have different number of centers
            codebook_centers = plydata.elements[-1]

            codebook_centers_torch = OrderedDict()
            codebook_centers_torch['features_dc'] = torch.from_numpy(np.asarray(codebook_centers['features_dc'], dtype=float_type)).cuda()
            codebook_centers_torch['features_rest'] = torch.from_numpy(np.concatenate(
                [
                    np.asarray(codebook_centers[f'features_rest_{i}'], dtype=float_type)[..., np.newaxis]
                    for i in range(max_coeffs_num)
                ], axis=1)).cuda()
            codebook_centers_torch['opacity'] = torch.from_numpy(np.asarray(codebook_centers['opacity'], dtype=float_type)).cuda()
            codebook_centers_torch['scaling'] = torch.from_numpy(np.asarray(codebook_centers['scaling'], dtype=float_type)).cuda()
            codebook_centers_torch['rotation_re'] = torch.from_numpy(np.asarray(codebook_centers['rotation_re'], dtype=float_type)).cuda()
            codebook_centers_torch['rotation_im'] = torch.from_numpy(np.asarray(codebook_centers['rotation_im'], dtype=float_type)).cuda()

            # If use half precision then we have to pointer cast the int16 to float16
            # and then cast them to floats, as that's the format that our renderer accepts
            if half_float:
                for k, v in codebook_centers_torch.items():
                    codebook_centers_torch[k] = pcast_i16_to_f32(v)

        # Iterate over the point clouds that are stored on top level of plyfile
        # to get the various fields values 
        for sh_degree in range(0, self.max_sh_degree+1):
            attribues_dict = self._parse_vertex_group(plydata.elements[sh_degree],
                                                      sh_degree,
                                                      float_type,
                                                      attribute_type,
                                                      max_coeffs_num,
                                                      quantised,
                                                      half_float,
                                                      codebook_centers_torch)

            xyz_list.append(attribues_dict['xyz'])
            features_dc_list.append(attribues_dict['features_dc'])
            features_rest_list.append(attribues_dict['features_rest'].transpose(1,2))
            opacity_list.append(attribues_dict['opacity'])
            scaling_list.append(attribues_dict['scaling'])
            rotation_list.append(attribues_dict['rotation'])
            degrees_list.append(attribues_dict['degrees'])

        # Concatenate the tensors into one, to be used in our program
        # TODO: allow multiple PCDs to be rendered/optimise and skip this step
        xyz = torch.cat((xyz_list), dim=0)
        features_dc = torch.cat((features_dc_list), dim=0)
        if not self.variable_sh_bands:
            features_rest = torch.cat((features_rest_list), dim=0)
        else:
            features_rest = features_rest_list
        opacity = torch.cat((opacity_list), dim=0)
        scaling = torch.cat((scaling_list), dim=0)
        rotation = torch.cat((rotation_list), dim=0)
        
        self._xyz = nn.Parameter(xyz.requires_grad_(True))
        self._features_dc = nn.Parameter(features_dc.requires_grad_(True))
        if not self.variable_sh_bands:
            self._features_rest = nn.Parameter(features_rest.requires_grad_(True))
        else:
            for tensor in features_rest_list:
                tensor.requires_grad_(True)
            self._features_rest = features_rest_list
        self._opacity = nn.Parameter(opacity.requires_grad_(True))
        self._scaling = nn.Parameter(scaling.requires_grad_(True))
        self._rotation = nn.Parameter(rotation.requires_grad_(True))
        self._degrees = torch.cat((degrees_list), dim=0)

        self.active_sh_degree = self.max_sh_degree
    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[valid_points_mask]
        self.xyz_gradient_accum_abs_max = self.xyz_gradient_accum_abs_max[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.max_weight = self.max_weight[valid_points_mask]
        if hasattr(self, '_degrees'):
            self._degrees = self._degrees[valid_points_mask]
    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs_max = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        max_weight = torch.zeros((new_xyz.shape[0]), device="cuda")
        self.max_weight = torch.cat((self.max_weight,max_weight),dim=0)

    def clone_points(self, clone_mask):
        new_xyz = self._xyz[clone_mask]
        new_features_dc = self._features_dc[clone_mask]
        new_features_rest = self._features_rest[clone_mask]    
        new_opacities = self._opacity[clone_mask]
        new_scaling = self._scaling[clone_mask]
        new_rotation = self._rotation[clone_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def split_points(self, split_mask, sigma=1.6, N=2):
        selected_pts_mask = split_mask
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / sigma)
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    # def densify_and_prune(self, clone_grad, split_grad, min_opacity, max_screen_size):
    #     grads = self.xyz_gradient_accum / self.denom
    #     grads[grads.isnan()] = 0.0
    #     grads = torch.norm(grads, dim=-1)

    #     padded_grad = torch.zeros((self.get_xyz.shape[0]), device="cuda")
    #     padded_grad[:grads.shape[0]] = grads.squeeze()
    #     split_mask = torch.logical_and(padded_grad >= split_grad, torch.max(self.get_scaling, dim=1).values > self.atom_scale)
    #     self.split_points(split_mask)
        

    #     padded_grad = torch.zeros((self.get_xyz.shape[0]), device="cuda")
    #     padded_grad[:grads.shape[0]] = grads.squeeze()
    #     clone_mask = padded_grad >= clone_grad
    #     self.clone_points(clone_mask)
    #     if max_screen_size:
    #         big_points_vs = self.max_radii2D > max_screen_size
    #         prune_mask = torch.logical_or(prune_mask, big_points_vs)

    #     prune_mask = torch.logical_or((self.get_opacity < min_opacity).squeeze(), torch.max(self.get_scaling, dim=1).values > 0.1 * self.spatial_lr_scale)
    #     self.prune_points(prune_mask)

    #     torch.cuda.empty_cache()
    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        # print("split:", new_xyz.shape[0])
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        # print("clone:", new_xyz.shape[0])
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, max_grad_abs, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        grads_abs = self.xyz_gradient_accum_abs / self.denom
        grads_abs[grads_abs.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads_abs, max_grad_abs, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()
    
    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        #TODO maybe use max instead of average
        # self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,2:], dim=-1, keepdim=True)

        self.xyz_gradient_accum_abs[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,2:], dim=-1, keepdim=True)
        self.xyz_gradient_accum_abs_max[update_filter] = torch.max(self.xyz_gradient_accum_abs_max[update_filter], torch.norm(viewspace_point_tensor.grad[update_filter,2:], dim=-1, keepdim=True))
        self.denom[update_filter] += 1

#add fun about reduce
    def _low_variance_colour_culling(self, threshold, weighted_variance, weighted_mean):
        original_degrees = torch.zeros_like(self._degrees)
        original_degrees.copy_(self._degrees)

        # Uniform colour culling
        weighted_colour_std = weighted_variance.sqrt()
        weighted_colour_std[weighted_colour_std.isnan()] = 0
        weighted_colour_std = weighted_colour_std.mean(dim=2).squeeze()

        std_mask = weighted_colour_std < threshold
        self._features_dc[std_mask] = (weighted_mean[std_mask] - 0.5) / 0.28209479177387814
        self._degrees[std_mask] = 0
        self._features_rest[std_mask] = 0

    def _low_distance_colour_culling(self, threshold, colour_distances):
        colour_distances[colour_distances.isnan()] = 0

        # Loop from active_sh_degree - 1 to 0, since the comparisons
        # are always done based on the max band that corresponds to active_sh_degree
        for sh_degree in range(self.active_sh_degree - 1, 0, -1):
            coeffs_num = (sh_degree+1)**2 - 1
            mask = colour_distances[:, sh_degree] < threshold
            self._degrees[mask] = torch.min(
                    torch.tensor([sh_degree], device="cuda", dtype=int),
                    self._degrees[mask]
                ).int()
            
            # Zero-out the associated SH coefficients for clarity,
            # as they won't be used in rasterisation due to the degrees field
            self._features_rest[mask, coeffs_num:] = 0

    def cull_sh_bands(self, cameras, threshold=0*np.sqrt(3)/255, std_threshold=0.):
        camera_positions = torch.stack([cam.camera_center for cam in cameras], dim=0)
        camera_viewmatrices = torch.stack([cam.world_view_transform for cam in cameras], dim=0)
        camera_projmatrices = torch.stack([cam.full_proj_transform for cam in cameras], dim=0)
        camera_fovx = torch.tensor([camera.FoVx for camera in cameras], device="cuda", dtype=torch.float32)
        camera_fovy = torch.tensor([camera.FoVy for camera in cameras], device="cuda", dtype=torch.float32)
        image_height = torch.tensor([camera.image_height for camera in cameras], device="cuda", dtype=torch.int32)
        image_width = torch.tensor([camera.image_width for camera in cameras], device="cuda", dtype=torch.int32)
        self._degrees = self.active_sh_degree * torch.ones((self.get_xyz.shape[0], 1), device="cuda", dtype=torch.int32)
        # Wrapping in a function since it's called with the same parameters twice
        def run_calculate_colours_variance():
            return calculate_colours_variance(
                camera_positions,
                self.get_xyz,
                self._opacity,
                self.get_scaling,
                self.get_rotation,
                camera_viewmatrices,
                camera_projmatrices,
                torch.tan(camera_fovx*0.5),
                torch.tan(camera_fovy*0.5),
                image_height,
                image_width,
                self.get_features,
                self._degrees,
                self.active_sh_degree,
                False)
        
        _, weighted_variance, weighted_mean = run_calculate_colours_variance()
        self._low_variance_colour_culling(std_threshold, weighted_variance, weighted_mean)

        # Recalculate to account for the changed values
        colour_distances, _, _ = run_calculate_colours_variance()
        self._low_distance_colour_culling(threshold, colour_distances)