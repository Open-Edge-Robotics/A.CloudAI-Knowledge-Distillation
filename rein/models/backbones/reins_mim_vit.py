import math
import numpy as np
import torch
from torch import nn
from mmengine.model.weight_init import xavier_init, constant_init
from mmcv.cnn import build_norm_layer

from mmseg.registry import MODELS
from .vision_transformer import TransformerEncoderLayer, VisionTransformer
from ..utils import (build_2d_sincos_position_embedding,
                     RelativePositionBias, trunc_normal_)

from .reins import Reins
from .utils import set_requires_grad, set_train


@MODELS.register_module()
class ReinsMIMVisionTransformer(VisionTransformer):
    """Vision Transformer for MIM-style model (Mask Image Modeling)
    classification (fine-tuning or linear probe).

    A PyTorch implement of : `An Image is Worth 16x16 Words: Transformers
    for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_

    Args:
        arch (str | dict): Vision Transformer architecture
            Default: 'b'
        img_size (int | tuple): Input image size
        patch_size (int | tuple): The patch size
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        drop_rate (float): Probability of an element to be zeroed.
            Defaults to 0.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        final_norm (bool): Whether to add a additional layer to normalize
            final feature map. Defaults to True.
        output_cls_token (bool): Whether output the cls_token. If set True,
            `with_cls_token` must be True. Defaults to True.
        interpolate_mode (str): Select the interpolate mode for position
            embeding vector resize. Defaults to "bicubic".
        patch_cfg (dict): Configs of patch embeding. Defaults to an empty dict.
        layer_cfgs (Sequence | dict): Configs of each transformer layer in
            encoder. Defaults to an empty dict.
        finetune (bool): Whether or not do fine-tuning. Defaults to True.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 arch='b',
                 img_size=224,
                 patch_size=16,
                 out_indices=-1,
                 use_window=True,
                 drop_rate=0,
                 drop_path_rate=0,
                 qkv_bias=True,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 final_norm=False,
                 output_cls_token=False,
                 interpolate_mode='bicubic',
                 init_values=0.0,
                 patch_cfg=dict(),
                 layer_cfgs=dict(),
                 finetune=True,
                 init_cfg=None,
                 reins_config=None,
                 **kwargs):
        super().__init__(arch,
                         img_size=img_size,
                         patch_size=patch_size,
                         out_indices=out_indices,
                         use_window=use_window,
                         drop_rate=drop_rate,
                         drop_path_rate=drop_path_rate,
                         qkv_bias=qkv_bias,
                         norm_cfg=norm_cfg,
                         final_norm=final_norm,
                         output_cls_token=output_cls_token,
                         interpolate_mode=interpolate_mode,
                         init_values=init_values,
                         patch_cfg=patch_cfg,
                         layer_cfgs=layer_cfgs,
                         init_cfg=init_cfg,
                         **kwargs)
        # if not self.final_norm:
        #     _, self.fc_norm = build_norm_layer(
        #         norm_cfg, self.embed_dims, postfix=1)
        self.reins: Reins = MODELS.build(reins_config)

        self.finetune = finetune
        if not self.finetune:
            self.frozen_stages = self.num_layers - 1  # all layers
            self._freeze_stages()

    def forward(self, x):
        B = x.shape[0]
        x, hw_shape = self.patch_embed(x)

        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.drop_after_pos(x)

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = self.reins.forward(
                x,
                i,
                batch_first=True,
                has_cls_token=True,
            )
            if i == len(self.layers) - 1 and self.final_norm:
                x = self.norm1(x)
            
            if i in self.out_indices:
                if self.with_cls_token:
                    # Remove class token and reshape token for decoder head
                    out = x[:, 1:]
                else:
                    out = x
                B, _, C = out.shape
                out = out.reshape(B, hw_shape[0], hw_shape[1],
                                  C).permute(0, 3, 1, 2).contiguous()
                if self.output_cls_token:
                    out = [out, x[:, 0]]
                outs.append(out)

        return self.reins.return_auto(outs)

    def train(self, mode: bool = True):
        if not mode:
            return super().train(mode)
        set_requires_grad(self, ["reins"])
        set_train(self, ["reins"])

    def state_dict(self, destination, prefix, keep_vars):
        state = super().state_dict(destination, prefix, keep_vars)
        keys = [k for k in state.keys() if "rein" not in k]
        for key in keys:
            state.pop(key)
            if key in destination:
                destination.pop(key)
        return state
        # if not self.final_norm:
        #     x = x[:, 1:, :].mean(dim=1)
        #     outs = self.fc_norm(x)
        # else:
        #     outs = x[:, 0]
        # return [outs]