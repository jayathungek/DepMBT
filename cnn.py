from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


#TODO
class MidFusionCNN(nn.Module):
    def __init__(self, dataset_const_namespace, embed_dim, linear_drop=0.1, num_layers=14, apply_augmentation=False):
        super().__init__()

#TODO
class ResBlock(nn.Module):
    """
    kernel_size: (k_temporal, k_spatial, k_filters)
    stride: (s_temporal, s_spatial)
    """
    def __init__(self, in_channels, out_channels, kernel_size, downsample):
        super().__init__()
        k_temporal, k_spatial= kernel_size
        p_temporal = 0 if k_temporal == 1 else 1
        p_spatial = 0 if k_spatial == 1 else 1
        self.shortcut = nn.Identity()

        self.conv1 = nn.Conv3d(in_channels, 
                                out_channels,
                                kernel_size=(k_temporal, k_spatial, k_spatial),
                                stride=1,
                                padding=(p_temporal, p_spatial, p_spatial))
        if downsample or (in_channels != out_channels):
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels,
                          out_channels,
                          kernel_size=(k_temporal, k_spatial, k_spatial),
                          stride=(1, 2, 2) if downsample else 1),
                nn.BatchNorm3d(out_channels)
            )
        self.bn1 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = nn.ReLU()(self.bn1(self.conv1(x)))
        x = x + shortcut
        return nn.ReLU()(x)


class TestResNet3d(nn.Module):
    def __init__(self, in_channels, resblock):
        super().__init__()

        self.layer1 = nn.Sequential(
            resblock(in_channels, 64, kernel_size=(1, 1), downsample=False),
            resblock(64, 64, kernel_size=(1, 3), downsample=False),
            resblock(64, 256, kernel_size=(1, 1), downsample=False)
        )
    
    def forward(self, x):
        x = self.layer1(x)
        return x

class AudioRes2D(nn.Module):
    def __init__(self, dataset_const_namespace, embed_dim, num_layers=3, apply_augmentation=False):
        super().__init__()

class VideoRes3D(nn.Module):
    def __init__(self, dataset_const_namespace, embed_dim, num_res_blocks=3, apply_augmentation=False):
        super().__init__()
        self.num_res_blocks = num_res_blocks
        self.apply_augmentation = apply_augmentation
        self.ds_constants = dataset_const_namespace
        self.embed_dim = embed_dim

class LateFusionCNN(nn.Module):
    def __init__(self, dataset_const_namespace, embed_dim, linear_drop=0.1, num_layers=14, apply_augmentation=False):
        super().__init__()

        

if __name__ == "__main__":
    inp = torch.rand(1, 64, 8, 56, 56)
    r3d = TestResNet3d(64, ResBlock)
    out = r3d(inp)
    print(out.shape)
    