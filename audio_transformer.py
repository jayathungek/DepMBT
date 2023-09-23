import time
import logging
from typing import Optional, Dict

import torch
import torch.nn as nn
from vision_transformer import VisionTransformer, Block
from timm.data import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

from helpers import to_2tuple
from patch_embed import PatchEmbed
from constants import *
from tokenizer import make_mfcc_input, make_rgb_input
from vitmbt import load_pretrained




class PretrainedAST(nn.Module):
    def __init__(self, pretrained_checkpoint_path, cutoff_layer, channels, model_name, no_class):
        super().__init__()
        print(f"Loading {model_name}...")
        t_start = time.time()
        cfg = self._cfg(pretrained_checkpoint_path)

        # EXPAND 1 CHANNEL SPECTROGRAM TO 3 CHANNELS WITH SAME CONTENT
        self.model = VisionTransformer(in_chans=channels, no_embed_class=no_class, img_size=(128, 800))
        load_pretrained(self.model, pretrained_cfg=cfg)
        delta_t = time.time() - t_start
        print(f"Loaded successfully in {delta_t:.1f}s")

        trimmed_layers = self.model.blocks[:cutoff_layer]
        del self.model.norm
        del self.model.fc_norm
        del self.model.head
        self.model.blocks = trimmed_layers

    def _cfg(self, url='', **kwargs):
        return {
            'url': url,
            'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
            'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
            'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
            'first_conv': 'patch_embed.proj', 'classifier': 'head',
            **kwargs
        }

    def forward(self, x):
        return self.model.forward_features(x)


if __name__ == "__main__":
    spec = make_mfcc_input(f"{DS_BASE}/{TEST_FILE}", SAMPLING_RATE_AS)
    # spec = make_rgb_input(f"{DS_BASE}/{TEST_FILE}", SAMPLING_RATE_AS)
    spec = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0)
    spec = spec.repeat(1, 3, 1, 1)
    print(spec.shape)
    ast = PretrainedAST(PRETRAINED_CHKPT, 12, 3, "audio layers", False)
    a = ast.model.embed_project(spec)
    a = ast.model.forward_features(a)
    print(a.shape)