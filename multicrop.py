from typing import Tuple
import torch
import torch.nn as nn
from torchvision import transforms as tfm

from constants import *


class MultiCrop(nn.Module):
    """ 
    Module for generating multiple global and local views of a 2D image or 3D Volume
    Input in the form B x H x W x ...
    """

    def __init__(
            self,
            image_size: Tuple,
            num_global_views: int=2,
            num_local_views: int=3,
            global_view_pct: float=0.7,
            local_view_pct: float=0.25
    ):
        super().__init__()
        self.image_height = image_size[0]
        self.image_width = image_size[1]
        self.num_global_views = num_global_views
        self.num_local_views = num_local_views
        self.global_view_height = int(self.image_height * GLOBAL_VIEW_PCT)
        self.global_view_width = int(self.image_width * GLOBAL_VIEW_PCT)

        self.local_view_height = int(self.image_height * LOCAL_VIEW_PCT)
        self.local_view_width = int(self.image_width * LOCAL_VIEW_PCT)

        # Could also try tfm.RandomResizedCrop to fill to image size instead of pad
        self.crop_global = tfm.RandomCrop(size=(self.global_view_height, self.global_view_width))
        self.crop_local = tfm.RandomCrop(size=(self.local_view_height, self.local_view_width))
        self.augment = tfm.Compose([
            tfm.ColorJitter(),
        ])


    def crop_and_pad(self, batch, local):
        if local:
            batch = self.crop_local(batch)
            height_diff = self.image_height - self.local_view_height
            width_diff = self.image_width - self.local_view_width
        else:
            batch = self.crop_global(batch)
            height_diff = self.image_height - self.global_view_height
            width_diff = self.image_width - self.global_view_width
        
        padding_left = padding_right = int(width_diff / 2)
        padding_top = padding_bottom = int(height_diff / 2)

        if width_diff % 2 != 0:
            padding_left += 1

        if height_diff % 2 != 0:
            padding_top += 1

        padding = (padding_left, padding_right, padding_top, padding_bottom)        
        padded_image = nn.ConstantPad2d(padding, 0)(batch)
        
        return padded_image
        

    def forward(self, batch):
        global_views = []
        local_views = []

        for _ in range(self.num_global_views):
            global_views.append(self.crop_and_pad(batch, local=False))

        for _ in range(self.num_local_views):
            local_views.append(self.crop_and_pad(batch, local=True))

        return global_views, local_views
