
""" Image to Patch Embedding using Conv2d

A convolution based approach to patchifying a 2D image w/ embedding projection.

Based on code in:
  * https://github.com/google-research/vision_transformer
  * https://github.com/google-research/big_vision/tree/main/big_vision

Hacked together by / Copyright 2020 Ross Wightman
"""
from typing import List, Optional, Callable

import torch.nn as nn

from helpers import to_2tuple

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(
            self,
            img_size: Optional[tuple] = (224, 224),
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer: Optional[Callable] = None,
            bias: bool = True,
    ):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        if img_size is not None:
            self.img_size = img_size
            self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
            self.num_patches = self.grid_size[0] * self.grid_size[1]
        else:
            self.img_size = None
            self.grid_size = None
            self.num_patches = None

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        if self.img_size is not None:
            assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
            assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")

        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        x = self.norm(x)
        return x
