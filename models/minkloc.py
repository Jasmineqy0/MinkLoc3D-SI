# Author: Jacek Komorowski
# Warsaw University of Technology
# Modified by: Kamil Zywanowski, Adam Banaszczyk, Michal Nowicki (Poznan University of Technology 2021)

#### ToDo: INCORPORATE POINTNETVLAD FEATURES ####
from models.pointnet.PointNetVlad import PointNetfeat
#################################################

import torch
import MinkowskiEngine as ME
from models.minkfpn import MinkFPN
from models.netvlad import MinkNetVladWrapper
import layers.pooling as pooling


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
