# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import Conv2d, build_activation_layer, build_norm_layer, ConvModule
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import MultiheadAttention
from mmengine.model import BaseModule, ModuleList, Sequential
from mmengine.model.weight_init import (constant_init, normal_init,
                                        trunc_normal_init)

from mmseg.models.utils import resize
from mmseg.registry import MODELS
from ..utils import PatchEmbed, nchw_to_nlc, nlc_to_nchw
from .utils import set_requires_grad, set_train
from .mit import MixFFN, EfficientMultiheadAttention, MixVisionTransformer, TransformerEncoderLayer


@MODELS.register_module()
class MixVisionTransformer_LoSA(MixVisionTransformer):
    def __init__(self, in_channels=3, embed_dims=64, num_stages=4, num_layers=[3, 4, 6, 3], num_heads=[1, 2, 4, 8], patch_sizes=[7, 3, 3, 3], strides=[4, 2, 2, 2], sr_ratios=[8, 4, 2, 1], out_indices=(0, 1, 2, 3), mlp_ratio=4, qkv_bias=True, drop_rate=0, attn_drop_rate=0, drop_path_rate=0, act_cfg=dict(type='GELU'), norm_cfg=dict(type='LN', eps=0.000001), pretrained=None, init_cfg=None, with_cp=False):
        super().__init__(in_channels, embed_dims, num_stages, num_layers, num_heads, patch_sizes, strides, sr_ratios, out_indices, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate, drop_path_rate, act_cfg, norm_cfg, pretrained, init_cfg, with_cp)

        # transformer encoder
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(num_layers))
        ]  # stochastic num_layer decay rule

        cur = 0
        self.layers = ModuleList()
        for i, num_layer in enumerate(num_layers):
            embed_dims_i = embed_dims * num_heads[i]
            patch_embed = PatchEmbed(
                in_channels=in_channels,
                embed_dims=embed_dims_i,
                kernel_size=patch_sizes[i],
                stride=strides[i],
                padding=patch_sizes[i] // 2,
                norm_cfg=norm_cfg)
            transformer_layer = TransformerEncoderLayer
            layer = ModuleList([
                transformer_layer(
                    embed_dims=embed_dims_i,
                    num_heads=num_heads[i],
                    feedforward_channels=mlp_ratio * embed_dims_i,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=dpr[cur + idx],
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    with_cp=with_cp,
                    sr_ratio=sr_ratios[i]
                    ) for idx in range(num_layer)
            ])
            in_channels = embed_dims_i
            # The ret[0] of build_norm_layer is norm name.
            norm = build_norm_layer(norm_cfg, embed_dims_i)[1]
            self.layers.append(ModuleList([patch_embed, layer, norm]))
            cur += num_layer


        r = 4
        self.adapt_LoSA_g = ModuleList()
        self.adapt_LoSA_x = ModuleList()
        for i in out_indices:
            # print(embed_dims * num_heads[i])
            w_a =  nn.Linear(embed_dims * num_heads[i], r, bias=False)
            w_b =  nn.Linear(r, embed_dims * num_heads[i], bias=False)
            gelu = nn.GELU()

            self.adapt_LoSA_g.append(nn.Sequential(w_a, gelu, w_b))

        for i in range(len(out_indices)):
            # print(embed_dims * num_heads[i])
            w_a =  nn.Linear(embed_dims * num_heads[out_indices[i-1]], r, bias=False)
            w_b =  nn.Linear(r, embed_dims * num_heads[out_indices[i]], bias=False)
            gelu = nn.GELU()

            self.adapt_LoSA_x.append(nn.Sequential(w_a, gelu, w_b))



    def set_cloud(self, mode = True):
        for i, layer in enumerate(self.layers):
            for block in layer[1]:
                block.encode_only = False

    def train(self, mode: bool = True):
        if not mode:
            return super().train(mode)
        set_requires_grad(self, ["adapt"])
        set_train(self, ["adapt"])

    def state_dict(self, destination, prefix, keep_vars):
        state = super().state_dict(destination, prefix, keep_vars)
        keys = [k for k in state.keys() if "adapt" not in k]
        for key in keys:
            state.pop(key)
            if key in destination:
                destination.pop(key)
        return state
    
    def forward(self, x):
        outs = self.forward_encode(x)
        outs = self.forward_adapt(outs)
        return outs

    def forward_encode(self, x):
        outs = []
        for i, layer in enumerate(self.layers):
            # print(i, layer)
            x, hw_shape = layer[0](x)
            for block in layer[1]:
                x = block(x, hw_shape)
            x = layer[2](x)
            x = nlc_to_nchw(x, hw_shape)
            if i in self.out_indices:
                outs.append(x)
                # print(x.shape)        
        return outs
    
    def forward_adapt(self, x):
        outs = []
        x_ = x[-1]
        # nchw_to_nlc, nlc_to_nchw
        for i, b_i in enumerate(x):
            hw_shape = b_i.size()[2:]
            # nchw_to_nlc
            b_i = nchw_to_nlc(b_i)
            x_i = nchw_to_nlc(resize(x_, hw_shape, mode = 'bilinear'))
            # print(hw_shape)
            # print(i, x_i.shape, b_i.shape)
            # print(self.adapt_LoSA_x[i])
            # print(self.adapt_LoSA_g[i])

            x_ = self.adapt_LoSA_x[i](x_i)
            x_ = self.adapt_LoSA_g[i](b_i + x_) + x_
            x_ = nlc_to_nchw(x_, hw_shape)

            outs.append(x_)        
        return outs
