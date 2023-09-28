import torch
import torch.nn as nn
from torchvision import transforms as tfm


class MultiCrop(nn.Module):
    """ 
    Module for generating multiple global and local views of a 2D image or 3D Volume
    Input in the form B x H x W x ...
    """

    def __init__(
            self,
            image_size: int=224,
            num_global_views: int=2,
            num_local_views: int=3,
            global_view_pct: float=0.7,
            local_view_pct: float=0.25
    ):
        super().__init__()
        self.image_size = image_size
        self.num_global_views = num_global_views
        self.num_local_views = num_local_views
        self.global_view_pct = global_view_pct
        self.local_view_pct = local_view_pct
        self.global_view_size = int(self.image_size * self.global_view_pct)
        self.local_view_size = int(self.image_size * self.local_view_pct)

        # Could also try tfm.RandomResizedCrop to fill to image size instead of pad
        self.crop_global = tfm.RandomCrop(size=self.global_view_size)
        self.crop_local = tfm.RandomCrop(size=self.local_view_size)
        self.augment = tfm.Compose([
            tfm.ColorJitter(),
        ])


    def crop_and_pad(self, batch, local):
        if local:
            batch = self.crop_local(batch)
            diff = self.image_size - self.local_view_size
        else:
            batch = self.crop_global(batch)
            diff = self.image_size - self.global_view_size
        
        padding_left = padding_right = padding_top = padding_bottom = int(diff / 2)
        if diff % 2 != 0:
            padding_left += 1
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
