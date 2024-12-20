# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from abc import ABCMeta, abstractmethod
from typing import List, Tuple

import torch
import torch.nn as nn
from .decode_head import BaseDecodeHead
from torch import Tensor
from mmseg.registry import MODELS

@MODELS.register_module()
class LinearHeadMS(BaseDecodeHead):
    """
    Linear Multi-Scale Head for DINOv2 ViT
    """
    def __init__(self, in_channels, **kwargs):
        super().__init__(in_channels=in_channels, **kwargs)
        # self.conv_cfg = [] # # # nn.Linear()
        # for i in range(self.out_channels):
        #     linear_layer = nn.Conv2d(self.channels, self.out_channels, kernel_size=1)
        #     self.conv_cfg.append(linear_layer)
        #     self.add_module('{}_layer'.format(i), linear_layer)
        
    def forward(self, inputs):
        """Forward function."""
        # print(inputs[0].shape)
        # print(inputs[0][:, :, 20, 20])
        # print(torch.randint(0, len(inputs), size=1))
        # layer_index = torch.randint(0, len(inputs), size=[1])
        # layer_index = 4
        # print(layer_index)
        # output = self.conv_cfg[layer_index](inputs[layer_index])
        # print(self.conv_seg.state_dict())
        # print(inputs[0].shape, output.shape, self.conv_seg)
        output = self.cls_seg(inputs)
        return output