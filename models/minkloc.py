# Author: Jacek Komorowski
# Warsaw University of Technology
# Modified by: Kamil Zywanowski, Adam Banaszczyk, Michal Nowicki (Poznan University of Technology 2021)

#### ToDo: INCORPORATE POINTNETVLAD FEATURES ####
from models.pointnet.PointNet import PointNetfeat_BfPooling, PointNetfeat_AfPooling
#################################################

import torch
from torch import nn
import torch.nn.functional as F
import MinkowskiEngine as ME
from models.minkfpn import MinkFPN
from models.netvlad import MinkNetVladWrapper
import layers.pooling as pooling

class PNT_GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, kernel=(4096,1)):
        super(PNT_GeM,self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps
        self.kernel = kernel

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), kernel_size=self.kernel).pow(1./p)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class MinkLoc(torch.nn.Module):
    def __init__(self, model, in_channels, feature_size, output_dim, planes, layers, num_top_down, conv0_kernel_size, num_points, combine_params):
        super().__init__()
        
        #### ToDo: INCORPORATE POINTNETVLAD FEATURES ####
        self.combine_params = combine_params
        self.num_points = num_points

        if (combine_params['with_pnt'] and combine_params['before_pooling']) or (combine_params['with_crosatt']):
            PNT_NUM_POINTS=num_points
            PNT_GLOBAL_FEAT=True
            PNT_FEATURE_TRANSFORM=True
            PNT_MAX_POOL= False
            
            # same as output_dim for pointnet only or first layer after conv for cross attention
            PNT_OUTPUT_DIM = feature_size if combine_params['with_pnt'] else planes[0]
                
            # added new args of PointNetfeat: output dim
            self.point_net = PointNetfeat_BfPooling(num_points=PNT_NUM_POINTS, global_feat=PNT_GLOBAL_FEAT,
                                                    feature_transform=PNT_FEATURE_TRANSFORM, max_pool=PNT_MAX_POOL,
                                                    output_dim=PNT_OUTPUT_DIM)
            if combine_params['with_pnt']:
                self.pnt_pooling = PNT_GeM()
            
        if combine_params['with_pnt'] and not combine_params['before_pooling']:
            PNT_NUM_POINTS=4096
            PNT_GLOBAL_FEAT=True
            PNT_FEATURE_TRANSFORM=True
            PNT_MAX_POOL= True # original value False
            PNT_OUTPUT_DIM = feature_size
            
            self.point_net = PointNetfeat_AfPooling(num_points=PNT_NUM_POINTS, global_feat=PNT_GLOBAL_FEAT,
                                                    feature_transform=PNT_FEATURE_TRANSFORM, max_pool=PNT_MAX_POOL,
                                                    output_dim=PNT_OUTPUT_DIM)
        #################################################        
        
        self.model = model
        self.in_channels = in_channels
        self.feature_size = feature_size    # Size of local features produced by local feature extraction block
        self.output_dim = output_dim        # Dimensionality of the global descriptor
        self.backbone = MinkFPN(in_channels=in_channels, out_channels=self.feature_size, num_top_down=num_top_down,
                                conv0_kernel_size=conv0_kernel_size, layers=layers, planes=planes,
                                combine_params=combine_params) # INCORPORATE POINTNETVLAD FEATURES
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
        
        # x = self.backbone(x)
        
        #### ToDo: INCORPORATE POINTNETVLAD FEATURES ####
        if self.combine_params['with_pnt'] or self.combine_params['with_crosatt']:
            PNT_NUM_POINTS = self.num_points

            PNT_x = batch['pnt_coords']
            
            if (self.combine_params['with_pnt'] and self.combine_params['before_pooling']) or self.combine_params['with_crosatt']:
                PNT_feats = self.point_net(PNT_x)
                # y = ME.SparseTensor(features=PNT_feats, coordinates=PNT_coords)
                
        if self.combine_params['with_crosatt']:
            
            PNT_x = PNT_x.squeeze(dim=1)
            PNT_x_list = [item for _, item in enumerate(PNT_x)]
            PNT_coords = ME.utils.batched_coordinates(PNT_x_list).to(PNT_x.device)
            
            x = self.backbone(x, PNT_coords, PNT_feats)
        else:
            x = self.backbone(x)
        #################################################          
        
        # x = self.backbone(x)

        # x is (num_points, n_features) tensor
        assert x.shape[1] == self.feature_size, 'Backbone output tensor has: {} channels. Expected: {}'.format(x.shape[1], self.feature_size)
        x = self.pooling(x)
        assert x.dim() == 2, 'Expected 2-dimensional tensor (batch_size,output_dim). Got {} dimensions.'.format(x.dim())
        assert x.shape[1] == self.output_dim, 'Output tensor has: {} channels. Expected: {}'.format(x.shape[1], self.output_dim)
        # x is (batch_size, output_dim) tensor
        
        #### ToDo: INCORPORATE POINTNETVLAD FEATURES ####
        if self.combine_params['with_pnt'] and not self.combine_params['before_pooling']:
            PNT_x, _ = self.point_net(PNT_x)
            
            # Combine Features of Pointnetvlad & MinkLoc3D-S
            if self.combine_params['with_way'] == 'add':
                x = x + PNT_x
                assert(x.shape[1] == self.output_dim)
            else:
                x = torch.cat((x, PNT_x), dim=1)
                assert(x.shape[1] == self.output_dim * 2)
        #################################################
        
        #### ToDo: INCORPORATE POINTNETVLAD FEATURES ####
        if self.combine_params['with_pnt'] and self.combine_params['before_pooling']:
            # # Combine Features of Pointnetvlad & MinkLoc3D-S
            # y = self.pooling(y)
            y = self.pnt_pooling(PNT_feats.view(-1, 4096, self.feature_size)).view(-1, self.feature_size)
            x = x + y
        #################################################
        
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
