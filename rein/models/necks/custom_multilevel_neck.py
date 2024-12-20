# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS


@MODELS.register_module()
class CustomMultiLevelNeck(nn.Module):
    """MultiLevelNeck.

    A neck structure connect vit backbone and decoder_heads.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        scales (List[float]): Scale factors for each input feature map.
            Default: [0.5, 1, 2, 4]
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer in ConvModule.
            Default: None.
    """

    def __init__(self, scales=[0.5, 1, 2, 4]):
        super().__init__()
        assert isinstance(scales, list)
        self.scales = scales
        self.num_outs = len(scales)

    def forward(self, ret):
        # assert len(inputs) == len(self.scales)
        if isinstance(ret[0], torch.Tensor):
            assert len(ret) == len(self.scales)
            for i in range(self.num_outs):
                scale_factor = self.scales[i]
                ret[i] = F.interpolate(
                    ret[i],
                    scale_factor=scale_factor,
                    mode='bilinear',
                    align_corners=False
                ) if scale_factor != 1 else ret[i]
        else:
            assert len(ret[0]) == len(self.scales)
            for i in range(self.num_outs):
                scale_factor = self.scales[i]
                ret[0][i] = F.interpolate(
                    ret[0][i],
                    scale_factor=scale_factor,
                    mode='bilinear',
                    align_corners=False
                ) if scale_factor != 1 else ret[0][i]

        return ret
