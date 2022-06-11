# Author: Jacek Komorowski
# Warsaw University of Technology

import torch.nn as nn
import torch
from torchtyping import TensorType
import MinkowskiEngine as ME
from MinkowskiEngine.modules.resnet_block import BasicBlock
from typing import List
from models.transformer.seq_manipulation import pad_sequence, unpad_sequences
from models.transformer.position_embedding import PositionEmbeddingCoordsSine
from models.transformer.transformers import TransformerCrossEncoderLayer, TransformerCrossEncoder
from models.resnet import ResNetBase

class MinkFPN(ResNetBase):
    # Feature Pyramid Network (FPN) architecture implementation using Minkowski ResNet building blocks
    def __init__(self, in_channels, out_channels, num_top_down=1, conv0_kernel_size=5, block=BasicBlock,
                 layers=(1, 1, 1), planes=(32, 64, 64), combine_params=None):
        assert len(layers) == len(planes)
        assert 1 <= len(layers)
        assert 0 <= num_top_down <= len(layers)
        self.num_bottom_up = len(layers)
        self.num_top_down = num_top_down
        self.conv0_kernel_size = conv0_kernel_size
        self.block = block
        self.layers = layers
        self.planes = planes
        self.lateral_dim = out_channels
        self.init_dim = planes[0]
        ResNetBase.__init__(self, in_channels, out_channels, D=3)
        #### ToDo: INCORPORATE POINTNETVLAD FEATURES ####
        self.with_crosatt  = combine_params['with_crosatt']
        if self.with_crosatt:
            d_embed = planes[0] # cross attention after first layer of conv
            
            nhead = combine_params['nhead']
            d_feedforward = combine_params['d_feedforward']
            dropout = combine_params['dropout']
            transformer_act = combine_params['transformer_act']
            pre_norm = combine_params['pre_norm']
            attention_type = combine_params['attention_type']
            sa_val_has_pos_emb = combine_params['sa_val_has_pos_emb']
            ca_val_has_pos_emb = combine_params['ca_val_has_pos_emb']
            num_encoder_layers = combine_params['num_encoder_layers']
            self.transformer_encoder_has_pos_emb = combine_params['transformer_encoder_has_pos_emb']
            
            self.pos_embed = PositionEmbeddingCoordsSine(3, d_embed, scale=1.0)
            
            encoder_layer = TransformerCrossEncoderLayer(
                d_embed, nhead, d_feedforward, dropout,
                activation=transformer_act,
                normalize_before=pre_norm,
                sa_val_has_pos_emb=sa_val_has_pos_emb,
                ca_val_has_pos_emb=ca_val_has_pos_emb,
                attention_type=attention_type,
            )
            encoder_norm = nn.LayerNorm(d_embed) if pre_norm else None
            self.transformer_encoder = TransformerCrossEncoder(
                encoder_layer, num_encoder_layers, encoder_norm,
                return_intermediate=False)
        #################################################

    def network_initialization(self, in_channels, out_channels, D):
        assert len(self.layers) == len(self.planes)
        assert len(self.planes) == self.num_bottom_up

        self.convs = nn.ModuleList()    # Bottom-up convolutional blocks with stride=2
        self.bn = nn.ModuleList()       # Bottom-up BatchNorms
        self.blocks = nn.ModuleList()   # Bottom-up blocks
        self.tconvs = nn.ModuleList()   # Top-down tranposed convolutions
        self.conv1x1 = nn.ModuleList()  # 1x1 convolutions in lateral connections

        # The first convolution is special case, with kernel size = 5
        self.inplanes = self.planes[0]
        self.conv0 = ME.MinkowskiConvolution(in_channels, self.inplanes, kernel_size=self.conv0_kernel_size,
                                             dimension=D)
        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)

        for plane, layer in zip(self.planes, self.layers):
            self.convs.append(ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D))
            self.bn.append(ME.MinkowskiBatchNorm(self.inplanes))
            self.blocks.append(self._make_layer(self.block, plane, layer))

        # Lateral connections
        for i in range(self.num_top_down):
            self.conv1x1.append(ME.MinkowskiConvolution(self.planes[-1 - i], self.lateral_dim, kernel_size=1,
                                                        stride=1, dimension=D))
            self.tconvs.append(ME.MinkowskiConvolutionTranspose(self.lateral_dim, self.lateral_dim, kernel_size=2,
                                                                stride=2, dimension=D))
        # There's one more lateral connection than top-down TConv blocks
        if self.num_top_down < self.num_bottom_up:
            # Lateral connection from Conv block 1 or above
            self.conv1x1.append(ME.MinkowskiConvolution(self.planes[-1 - self.num_top_down], self.lateral_dim, kernel_size=1,
                                                        stride=1, dimension=D))
        else:
            # Lateral connection from Con0 block
            self.conv1x1.append(ME.MinkowskiConvolution(self.planes[0], self.lateral_dim, kernel_size=1,
                                                        stride=1, dimension=D))

        self.relu = ME.MinkowskiReLU(inplace=True)
        
#### ToDo: INCORPORATE POINTNETVLAD FEATURES ####
    def batch_feat_size(self,x) -> List[int]:
        _, batch_feat_size = torch.unique(x[:,0], return_counts=True)
        return batch_feat_size.tolist()
    
    def batch_tolist(self, x:TensorType, seq:List[int]) -> List[TensorType]:
        x = list(torch.split(x, seq))
        return x
    
    def combine_cross_attention(self, x, y_c, y_f):
        x_batch_feat_size = self.batch_feat_size(x.C)
        y_batch_feat_size = self.batch_feat_size(y_c)
        
        x_pe = self.batch_tolist(self.pos_embed(x.C[:, 1:]), x_batch_feat_size)
        y_pe = self.batch_tolist(self.pos_embed(y_c[:, 1:]), y_batch_feat_size)
        y_feats_un = self.batch_tolist(y_f, y_batch_feat_size)
        x_feats_un = self.batch_tolist(x.F, x_batch_feat_size)

        x_pe_padded, _, _ = pad_sequence(x_pe)
        y_pe_padded, _, _ = pad_sequence(y_pe)
        
        x_feats_padded, x_key_padding_mask, _ = pad_sequence(x_feats_un,
                                                                require_padding_mask=True)
        y_feats_padded, y_key_padding_mask, _ = pad_sequence(y_feats_un,
                                                                require_padding_mask=True)
        
        x_feats_cond, y_feats_cond = self.transformer_encoder(
            x_feats_padded, y_feats_padded,
            src_key_padding_mask=x_key_padding_mask,
            tgt_key_padding_mask=y_key_padding_mask,
            src_pos=x_pe_padded if self.transformer_encoder_has_pos_emb else None,
            tgt_pos=y_pe_padded if self.transformer_encoder_has_pos_emb else None,
        )
        
        x_feats_cond = torch.squeeze(x_feats_cond, dim=0)
        y_feats_cond = torch.squeeze(y_feats_cond, dim=0)
        x_feats_list = unpad_sequences(x_feats_cond, x_batch_feat_size)
        # y_feats_list = unpad_sequences(y_feats_cond, y_batch_feat_size)
        
        x_feats = torch.vstack(x_feats_list)
        # y_feats = torch.vstack(y_feats_list)
        
        # x =  ME.SparseTensor(coordinates=x.C, features=x_feats)
        # y =  ME.SparseTensor(coordinates=y.C, features=y_feats, coordinate_manager=x.coordinate_manager)
        # x = x + y
        x = ME.SparseTensor(coordinates=x.C, features=x_feats)
        # x.F = x_feats
        
        return x
#################################################

    def forward(self, x, y_c=None, y_f=None):
        # *** BOTTOM-UP PASS ***
        # First bottom-up convolution is special (with bigger stride)
        feature_maps = []
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu(x)
        if self.num_top_down == self.num_bottom_up:
            feature_maps.append(x)

        #### ToDo: INCORPORATE POINTNETVLAD FEATURES ####
        if self.with_crosatt:
            x = self.combine_cross_attention(x, y_c, y_f)
        #################################################
        
        # BOTTOM-UP PASS
        for ndx, (conv, bn, block) in enumerate(zip(self.convs, self.bn, self.blocks)):
            x = conv(x)     # Decreases spatial resolution (conv stride=2)
            x = bn(x)
            x = self.relu(x)
            x = block(x) # after here results differ each time
            if self.num_bottom_up - 1 - self.num_top_down <= ndx < len(self.convs) - 1:
                feature_maps.append(x)

        assert len(feature_maps) == self.num_top_down

        x = self.conv1x1[0](x)

        # TOP-DOWN PASS
        for ndx, tconv in enumerate(self.tconvs):
            x = tconv(x)        # Upsample using transposed convolution
            x = x + self.conv1x1[ndx+1](feature_maps[-ndx - 1])

        return x
