# Author: Jacek Komorowski
# Warsaw University of Technology
# Modified by: Kamil Zywanowski, Adam Banaszczyk, Michal Nowicki (Poznan University of Technology 2021)

#### ToDo: INCORPORATE POINTNETVLAD FEATURES ####
from __future__ import print_function
import torch

torch.cuda.empty_cache()

from numba import njit, jit
#################################################

import MinkowskiEngine as ME

from models.minkfpn import MinkFPN
from models.netvlad import MinkNetVladWrapper
import layers.pooling as pooling

#### ToDo: INCORPORATE POINTNETVLAD FEATURES ####

import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import math


class Flatten(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, input):
        return input.view(input.size(0), -1)


class STN3d(nn.Module):
    def __init__(self, num_points=2500, k=3, use_bn=True):
        super(STN3d, self).__init__()
        self.k = k
        self.kernel_size = 3 if k == 3 else 1
        self.channels = 1 if k == 3 else k
        self.num_points = num_points
        self.use_bn = use_bn
        self.conv1 = torch.nn.Conv2d(self.channels, 64, (1, self.kernel_size))
        self.conv2 = torch.nn.Conv2d(64, 128, (1,1))
        self.conv3 = torch.nn.Conv2d(128, 1024, (1,1))
        self.mp1 = torch.nn.MaxPool2d((num_points, 1), 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.fc3.weight.data.zero_()
        self.fc3.bias.data.zero_()
        self.relu = nn.ReLU()

        if use_bn:
            self.bn1 = nn.BatchNorm2d(64)
            self.bn2 = nn.BatchNorm2d(128)
            self.bn3 = nn.BatchNorm2d(1024)
            self.bn4 = nn.BatchNorm1d(512)
            self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        if self.use_bn:
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
        else:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
        x = self.mp1(x)
        x = x.view(-1, 1024)

        if self.use_bn:
            x = F.relu(self.bn4(self.fc1(x)))
            x = F.relu(self.bn5(self.fc2(x)))
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).astype(np.float32))).view(
            1, self.k*self.k).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

@njit
def to_spherical_me(idx, points, dataset_name):
    spherical_points = []
    for point in points:
        # if (np.abs(point[:3]) < 1e-4).all():
        #     continue
        r = np.linalg.norm(point[:3])

        # Theta is calculated as an angle measured from the y-axis towards the x-axis
        # Shifted to range (0, 360)
        theta = np.arctan2(point[1], point[0]) * 180 / np.pi
        if theta < 0:
            theta += 360

        if dataset_name == "USyd":
            # VLP-16 has 2 deg VRes and (+15, -15 VFoV).
            # Phi calculated from the vertical axis, so (75, 105)
            # Shifted to (0, 30)
            phi = (np.arccos(point[2] / r) * 180 / np.pi) - 75

        elif dataset_name in ['IntensityOxford', 'Oxford']:
            # Oxford scans are built from a 2D scanner.
            # Phi calculated from the vertical axis, so (0, 180)
            phi = np.arccos(point[2] / r) * 180 / np.pi

        elif dataset_name == ['KITTI', 'TUM']:
            # HDL-64 has 0.4 deg VRes and (+2, -24.8 VFoV).
            # Phi calculated from the vertical axis, so (88, 114.8)
            # Shifted to (0, 26.8)
            phi = (np.arccos(point[2] / r) * 180 / np.pi) - 88

        if point.shape[-1] == 4:
            spherical_points.append([idx, r, theta, phi, point[3]])
        else:
            spherical_points.append([idx, r, theta, phi])

    return spherical_points


class PointNetfeat(nn.Module):
    def __init__(self, num_points=2500, global_feat=True, feature_transform=False, max_pool=True, feature_size=1024):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d(num_points=num_points, k=3, use_bn=False)
        self.feature_trans = STN3d(num_points=num_points, k=64, use_bn=False)
        self.apply_feature_trans = feature_transform
        self.conv1 = torch.nn.Conv2d(1, 64, (1, 3))
        self.conv2 = torch.nn.Conv2d(64, 64, (1, 1))
        self.conv3 = torch.nn.Conv2d(64, 64, (1, 1))
        self.conv4 = torch.nn.Conv2d(64, 128, (1, 1))
        # self.conv5 = torch.nn.Conv2d(128, 1024, (1, 1))
        #### ToDo: INCORPORATE POINTNETVLAD FEATURES ####
        self.conv5 = torch.nn.Conv2d(128, feature_size, (1, 1))
        self.feature_size = feature_size
        #################################################
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        # self.bn5 = nn.BatchNorm2d(1024)
        #### ToDo: INCORPORATE POINTNETVLAD FEATURES ####
        self.bn5 = nn.BatchNorm2d(feature_size)
        #################################################
        self.mp1 = torch.nn.MaxPool2d((num_points, 1), 1)
        self.num_points = num_points
        self.global_feat = global_feat
        self.max_pool = max_pool

    def forward(self, x):
        batchsize = x.size()[0]
        #### ToDo: INCORPORATE POINTNETVLAD FEATURES ####
        coords = []
        for idx, e in enumerate(np.squeeze(x, axis=1)):
            # Convert coordinates to spherical, return [batch_idx, r, theta, phi] with added batch_idx for later conversion of sparse tensor
            spherical_e_me = torch.tensor(to_spherical_me(idx, torch.Tensor.cpu(e).numpy(), 'TUM'), dtype=torch.int32)
            coords.append(spherical_e_me)
        coords = torch.vstack(coords)
        #################################################
        trans = self.stn(x)
        x = torch.matmul(torch.squeeze(x), trans)
        x = x.view(batchsize, 1, -1, 3)
        #x = x.transpose(2,1)
        #x = torch.bmm(x, trans)
        #x = x.transpose(2,1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        pointfeat = x
        if self.apply_feature_trans:
            f_trans = self.feature_trans(x)
            x = torch.squeeze(x)
            if batchsize == 1:
                x = torch.unsqueeze(x, 0)
            x = torch.matmul(x.transpose(1, 2), f_trans)
            x = x.transpose(1, 2).contiguous()
            x = x.view(batchsize, 64, -1, 1)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.bn5(self.conv5(x))
        if not self.max_pool:
            #### ToDo: INCORPORATE POINTNETVLAD FEATURES ####
            feats = x.view(-1, self.feature_size)
            # return a sparse tensor with pointnet features of all points
            return coords, feats
            #################################################
            return x
        else:
            x = self.mp1(x)
            x = x.view(-1, 1024)
            if self.global_feat:
                return x, trans
            else:
                x = x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
                return torch.cat([x, pointfeat], 1), trans

#################################################


class MinkLoc(torch.nn.Module):
    def __init__(self, model, in_channels, feature_size, output_dim, planes, layers, num_top_down, conv0_kernel_size, combine_pnt, combine_way, cross_att_pnt):
        super().__init__()
        
        #### ToDo: INCORPORATE POINTNETVLAD FEATURES ####
        self.combine_pnt = combine_pnt
        self.combine_way = combine_way
        self.cross_att_pnt = cross_att_pnt

        if combine_pnt:
            PNT_NUM_POINTS=4096
            PNT_GLOBAL_FEAT=True
            PNT_FEATURE_TRANSFORM=True
            # original value False
            PNT_MAX_POOL= False
            
            # add new args of PointNetfeat: feature_size
            self.point_net = PointNetfeat(num_points=PNT_NUM_POINTS, global_feat=PNT_GLOBAL_FEAT,
                                        feature_transform=PNT_FEATURE_TRANSFORM, max_pool=PNT_MAX_POOL,
                                        feature_size=feature_size)
        elif cross_att_pnt:
            PNT_NUM_POINTS=4096
            PNT_GLOBAL_FEAT=True
            PNT_FEATURE_TRANSFORM=True
            # original value False
            PNT_MAX_POOL= False
            CROSS_FEATURE_SIZE = 32
            
            self.point_net = PointNetfeat(num_points=PNT_NUM_POINTS, global_feat=PNT_GLOBAL_FEAT,
                                        feature_transform=PNT_FEATURE_TRANSFORM, max_pool=PNT_MAX_POOL,
                                        feature_size=CROSS_FEATURE_SIZE)
            
        #################################################        
        
        self.model = model
        self.in_channels = in_channels
        self.feature_size = feature_size    # Size of local features produced by local feature extraction block
        self.output_dim = output_dim        # Dimensionality of the global descriptor
        self.backbone = MinkFPN(in_channels=in_channels, out_channels=self.feature_size, num_top_down=num_top_down,
                                conv0_kernel_size=conv0_kernel_size, layers=layers, planes=planes,
                                cross_att_pnt=cross_att_pnt) # INCORPORATE POINTNETVLAD FEATURES
        self.n_backbone_features = output_dim

        if model == 'MinkFPN_Max':
            assert self.feature_size == self.output_dim, 'output_dim must be the same as feature_size'
            self.pooling = pooling.MAC()
        elif model == 'MinkFPN_GeM':
            assert self.feature_size == self.output_dim, 'output_dim must be the same as feature_size'
            self.pooling = pooling.GeM()
        elif model == 'MinkFPN_NetVlad':
            self.pooling = MinkNetVladWrapper(feature_size=self.feature_size, output_dim=self.output_dim,
                                              cluster_size=64, gating=False)
        elif model == 'MinkFPN_NetVlad_CG':
            self.pooling = MinkNetVladWrapper(feature_size=self.feature_size, output_dim=self.output_dim,
                                              cluster_size=64, gating=True)
        else:
            raise NotImplementedError('Model not implemented: {}'.format(model))

    def forward(self, batch):
        # Coords must be on CPU, features can be on GPU - see MinkowskiEngine documentation
        feats = batch['features']
        feats = feats.to('cuda')
        coords = batch['coords']
        coords = coords.to('cuda')

        x = ME.SparseTensor(feats, coords)
        
        #### ToDo: INCORPORATE POINTNETVLAD FEATURES ####
        if self.cross_att_pnt or self.combine_pnt:
            PNT_NUM_POINTS = 4096

            PNT_x = batch['clouds']
            PNT_x = PNT_x.to('cuda')
            
            PNT_x = PNT_x.view((-1, 1, PNT_NUM_POINTS, 3))
            
            PNT_coords, PNT_feats = self.point_net(PNT_x)
            PNT_coords = PNT_coords.to('cuda')
            PNT_feats = PNT_feats.to('cuda')
            y = ME.SparseTensor(features=PNT_feats, coordinates=PNT_coords, 
                    coordinate_manager=x.coordinate_manager)
            if self.cross_att_pnt:
                x = self.backbone(x, y)
            else:
                x = self.backbone(x)
        else:
            x = self.backbone(x)
        #################################################        
        
        # x = self.backbone(x)

        #### ToDo: INCORPORATE POINTNETVLAD FEATURES ####
        if self.combine_pnt:
            # PNT_NUM_POINTS = 4096

            # PNT_x = batch['clouds']
            # PNT_x = PNT_x.to('cuda')
            
            # PNT_x = PNT_x.view((-1, 1, PNT_NUM_POINTS, 3))
            
            # PNT_coords, PNT_feats = self.point_net(PNT_x)
            # PNT_coords = PNT_coords.to('cuda')
            # PNT_feats = PNT_feats.to('cuda')
            
            y = ME.SparseTensor(features=PNT_feats, coordinates=PNT_coords, 
                                coordinate_manager=x.coordinate_manager)
            
            # Combine Features of Pointnetvlad & MinkLoc3D-S
            assert self.combine_way == 'cat', 'concat features only'
            x = x + y
        #################################################

        # x is (num_points, n_features) tensor
        assert x.shape[1] == self.feature_size, 'Backbone output tensor has: {} channels. Expected: {}'.format(x.shape[1], self.feature_size)
        x = self.pooling(x)
        assert x.dim() == 2, 'Expected 2-dimensional tensor (batch_size,output_dim). Got {} dimensions.'.format(x.dim())
        assert x.shape[1] == self.output_dim, 'Output tensor has: {} channels. Expected: {}'.format(x.shape[1], self.output_dim)
        # x is (batch_size, output_dim) tensor
        
        
        
        return x

    def print_info(self):
        print('Model class: MinkLoc')
        n_params = sum([param.nelement() for param in self.parameters()])
        print('Total parameters: {}'.format(n_params))
        n_params = sum([param.nelement() for param in self.backbone.parameters()])
        print('Backbone parameters: {}'.format(n_params))
        n_params = sum([param.nelement() for param in self.pooling.parameters()])
        print('Aggregation parameters: {}'.format(n_params))
        if hasattr(self.backbone, 'print_info'):
            self.backbone.print_info()
        if hasattr(self.pooling, 'print_info'):
            self.pooling.print_info()
