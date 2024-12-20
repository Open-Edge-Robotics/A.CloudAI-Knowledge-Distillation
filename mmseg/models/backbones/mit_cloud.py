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

from mmseg.registry import MODELS
from ..utils import PatchEmbed, nchw_to_nlc, nlc_to_nchw
from .utils import set_requires_grad, set_train
from .mit import MixFFN, EfficientMultiheadAttention, MixVisionTransformer, TransformerEncoderLayer


class TransformerEncoderLayer_cloud(BaseModule):
    """Implements one encoder layer in Segformer.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        drop_rate (float): Probability of an element to be zeroed.
            after the feed forward layer. Default 0.0.
        attn_drop_rate (float): The drop out rate for attention layer. 
            Default 0.0.
        drop_path_rate (float): stochastic depth rate. Default 0.0.
        qkv_bias (bool): enable bias for qkv if True.
            Default: True.
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default: False.
        init_cfg (dict, optional): Initialization config dict.
            Default:None.
        sr_ratio (int): The ratio of spatial reduction of Efficient Multi-head
            Attention of Segformer. Default: 1.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 qkv_bias=True,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 batch_first=True,
                 sr_ratio=1,
                 with_cp=False,
                 encode_only = False):
        super().__init__()
        down_size = 64
        # The ret[0] of build_norm_layer is norm name.
        self.encode_only = encode_only
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.attn = EfficientMultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            batch_first=batch_first,
            qkv_bias=qkv_bias,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratio)

        # The ret[0] of build_norm_layer is norm name.
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.ffn = MixFFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg)

        self.with_cp = with_cp

        self.adaptformer_down_proj = nn.Linear(embed_dims, down_size)
        self.non_linear_func = nn.ReLU()
        self.adaptformer_up_proj = nn.Linear(down_size, embed_dims)
        self.adaptformer_layer_norm_before = nn.LayerNorm(embed_dims)

        self.adaptformer_decoder = nn.Sequential(
            # nn.ReLU(),
            # build_norm_layer(norm_cfg, embed_dims)[1],
            nn.Linear(embed_dims, 384),
            nn.ReLU(),
            build_norm_layer(norm_cfg, 384)[1],
            nn.Linear(384, 384),

        )

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.adaptformer_down_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.adaptformer_up_proj.weight)
            nn.init.zeros_(self.adaptformer_down_proj.bias)
            nn.init.zeros_(self.adaptformer_up_proj.bias)
    
    def forward_adapt(self, x):
        down = self.adaptformer_down_proj(x)
        # down = self.adaptformer_layer_norm_before(down)
        down = self.non_linear_func(down)
        up = self.adaptformer_up_proj(down)
        up = self.adaptformer_layer_norm_before(up)
        return up
    
    def forward(self, x, hw_shape):
        def _inner_forward(x):
            x = self.attn(self.norm1(x), hw_shape, identity=x)
            x_ = self.forward_adapt(x)
            x = self.ffn(self.norm2(x), hw_shape, identity=x)
            x = x + x_
            if self.encode_only:
                return x
            x_decode = self.adaptformer_decoder(x)
            x_decode = nlc_to_nchw(x_decode, hw_shape)
            return x, x_decode

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x

@MODELS.register_module()
class MixVisionTransformer_clouddevice(MixVisionTransformer):
    def __init__(self, is_cloud = False, in_channels=3, embed_dims=64, num_stages=4, num_layers=[3, 4, 6, 3], num_heads=[1, 2, 4, 8], patch_sizes=[7, 3, 3, 3], strides=[4, 2, 2, 2], sr_ratios=[8, 4, 2, 1], out_indices=(0, 1, 2, 3), mlp_ratio=4, qkv_bias=True, drop_rate=0, attn_drop_rate=0, drop_path_rate=0, act_cfg=dict(type='GELU'), norm_cfg=dict(type='LN', eps=0.000001), pretrained=None, init_cfg=None, with_cp=False):
        super().__init__(in_channels, embed_dims, num_stages, num_layers, num_heads, patch_sizes, strides, sr_ratios, out_indices, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate, drop_path_rate, act_cfg, norm_cfg, pretrained, init_cfg, with_cp)

        self.is_cloud = is_cloud
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
            transformer_layer = TransformerEncoderLayer_cloud
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
                    sr_ratio=sr_ratios[i],
                    encode_only=False if self.is_cloud else True) for idx in range(num_layer)
            ])
            in_channels = embed_dims_i
            # The ret[0] of build_norm_layer is norm name.
            norm = build_norm_layer(norm_cfg, embed_dims_i)[1]
            self.layers.append(ModuleList([patch_embed, layer, norm]))
            cur += num_layer

        # self.adapter_decoder_conv = nn.ModuleList()
        # for i, num_layer in enumerate(num_layers):
        #     embed_dims_i = embed_dims * num_heads[i]
        #     for j in range(num_layer):
        #         self.adapter_decoder_conv.append(
        #             ConvModule(
        #                 in_channels=embed_dims_i,
        #                 out_channels=embed_dims,
        #                 kernel_size=1,
        #                 stride=1,
        #                 norm_cfg=None,
        #                 act_cfg=act_cfg))
            
        # self.adapter_fusion_conv = ConvModule(
        #     in_channels=embed_dims * cur,
        #     out_channels=384,
        #     kernel_size=1,
        #     norm_cfg=None)

    def set_cloud(self, mode = True):
        for i, layer in enumerate(self.layers):
            for block in layer[1]:
                block.encode_only = False

    def train(self, mode: bool = True):
        if not mode:
            return super().train(mode)
        set_requires_grad(self, ["adaptformer"])
        set_train(self, ["adaptformer"])

    def state_dict(self, destination, prefix, keep_vars):
        state = super().state_dict(destination, prefix, keep_vars)
        keys = [k for k in state.keys() if "adaptformer" not in k]
        for key in keys:
            state.pop(key)
            if key in destination:
                destination.pop(key)
        return state
    
    def forward(self, x):
        if self.is_cloud:
            outs = self.forward_decode(x)
        else:
            outs = self.forward_encode(x)
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
        return outs
    
    def forward_decode(self, x):
        outs = []
        outs_decode = []

        for i, layer in enumerate(self.layers):
            # print(i, layer)
            x, hw_shape = layer[0](x)
            for block in layer[1]:
                x, x_decode = block(x, hw_shape)
                outs_decode.append(x_decode)
            x = layer[2](x)
            x = nlc_to_nchw(x, hw_shape)
            if i in self.out_indices:
                outs.append(x)
                #outs_decode.append(x_decode)

        return outs, outs_decode