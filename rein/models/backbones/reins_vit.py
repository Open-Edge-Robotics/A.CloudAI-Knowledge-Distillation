import torch

from mmseg.models.builder import BACKBONES, MODELS
from mmseg.models.backbones.vit import VisionTransformer

from .reins import Reins
from .utils import set_requires_grad, set_train


@BACKBONES.register_module()
class ReinsVisionTransformer(VisionTransformer):
    def __init__(
        self,
        reins_config=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.reins: Reins = MODELS.build(reins_config)

    # def forward_features(self, x, masks=None):
    #     B, _, h, w = x.shape
    #     H, W = h // self.patch_size, w // self.patch_size
    #     x = self.prepare_tokens_with_masks(x, masks)
    #     outs = []
    #     for idx, blk in enumerate(self.blocks):
    #         x = blk(x)
    #         x = self.reins.forward(
    #             x,
    #             idx,
    #             batch_first=True,
    #             has_cls_token=True,
    #         )
    #         if idx in self.out_indices:
    #             outs.append(
    #                 x[:, 1:, :].permute(0, 2, 1).reshape(B, -1, H, W).contiguous()
    #             )
    #     return self.reins.return_auto(outs)

    def forward(self, inputs):
        B = inputs.shape[0]

        x, hw_shape = self.patch_embed(inputs)

        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self._pos_embeding(x, hw_shape, self.pos_embed)

        if not self.with_cls_token:
            # Remove class token for transformer encoder input
            x = x[:, 1:]

        if self.pre_norm:
            x = self.pre_ln(x)

        outs = []
        if self.out_origin:
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

        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = self.reins.forward(
                x,
                i,
                batch_first=True,
                has_cls_token=True,
            )
            if i == len(self.layers) - 1:
                if self.final_norm:
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