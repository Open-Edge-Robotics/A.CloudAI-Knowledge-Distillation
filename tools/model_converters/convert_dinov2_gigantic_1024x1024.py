import torch
import os.path as osp
from collections import OrderedDict
from torch import Tensor
import torch.nn.functional as F
import sys


def load_backbone(path):
    if not osp.isfile(path):
        raise FileNotFoundError(
            f"{path} dont exist(absolute path: {osp.abspath(path)})"
        )
    weight = torch.load(path, map_location="cpu")

    new_weight=dict()

    print(weight["pos_embed"].shape)
    new_weight["pos_embed"] = torch.cat(
        (
            weight["pos_embed"][:, :1, :],
            F.interpolate(
                weight["pos_embed"][:, 1:, :]
                .reshape(1, 37, 37, 1536)
                .permute(0, 3, 1, 2),
                size=(64, 64),
                mode="bicubic",
                align_corners=False,
            )
            .permute(0, 2, 3, 1)
            .reshape(1, 4096, 1536),
        ),
        dim=1,
    )
    new_weight["patch_embed.proj.weight"] = F.interpolate(
        weight["patch_embed.proj.weight"].float(),
        size=(16, 16),
        mode="bicubic",
        align_corners=False,
    )

    for key in weight:
        print(key)
        if 'w12' in key:
            new_weight[key.replace('w12', 'fc1')] = weight[key]
        if 'w3' in key:
            new_weight[key.replace('w3', 'fc2')] = weight[key]
    return new_weight


def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <path> <save_path>")
        sys.exit(1)
    path = sys.argv[1]
    save_path = sys.argv[2]
    state = load_backbone(path)
    torch.save(state, save_path)


# Check if the script is run directly (and not imported)
if __name__ == "__main__":
    main()