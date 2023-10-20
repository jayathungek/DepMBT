import io
import math
import sys
import csv
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Callable, Union, List
from types import ModuleType

import librosa
import ffmpeg
import numpy as np
import torch
from torch import nn
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image

from timm.layers.format import Format, nchw_to
from timm.layers.helpers import to_2tuple
from face import CropFace
import librosa
import librosa.display as lrd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from constants import *


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    output_fmt: Format

    def __init__(
            self,
            img_size: Union[int, tuple] = 224,
            patch_size: int = 16,
            num_frames: int = 10,
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer: Optional[Callable] = None,
            flatten: bool = True,
            output_fmt: Optional[str] = None,
            bias: bool = True,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        if output_fmt is not None:
            self.flatten = False
            self.output_fmt = Format(output_fmt)
        else:
            # flatten spatial dim and transpose to channels last, kept for bwd compat
            self.flatten = flatten
            self.output_fmt = Format.NCHW

        self.proj = nn.Conv2d(in_chans * num_frames, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]})."
        assert W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        elif self.output_fmt != Format.NCHW:
            x = nchw_to(x, self.output_fmt)
        x = self.norm(x)
        return x


def normalize(arr: np.ndarray):
    return arr / arr.max()


def get_rgb_frames(video_path: str, video_length: float, ensure_frames_len: int=None) -> np.array:
    min_resize = 256
    new_width = "(iw/min(iw,ih))*{}".format(min_resize)
    cmd = (
        ffmpeg
        .input(video_path)
        .trim(start=0, end=10)
        .filter("fps", fps=math.ceil(FRAMES/video_length))
        .filter("scale", new_width, -1)
        .output("pipe:", format="image2pipe")
    )
    jpeg_bytes, _ = cmd.run(capture_stdout=True, quiet=True)
    jpeg_bytes = jpeg_bytes.split(JPEG_HEADER)[1:]
    jpeg_bytes = map(lambda x: JPEG_HEADER + x, jpeg_bytes)
    all_frames = list(jpeg_bytes)
    if ensure_frames_len is not None:
        if len(all_frames) > ensure_frames_len:
            all_frames = all_frames[:ensure_frames_len]
        elif len(all_frames) < ensure_frames_len:
            # repeat last frame by the difference between target length and actual length
            diff = ensure_frames_len - len(all_frames)
            all_frames = all_frames + [all_frames[-1] for _ in range(diff)]
            if DEBUG:
                print(f"get_rgb_frames: appended {diff} repeated frames to {video_path}")
    return all_frames

def save_video_frames_bytes(frames: List[bytearray], save_dir: str):
    newdir = Path(save_dir)
    newdir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        with open(newdir / f"{i}.jpeg", "wb") as fh:
            frame = frame / 255.0
            fh.write(frame)

def save_video_frames_tensors(frames: torch.tensor, save_dir: str, rows: int=2):
    save_image(frames, fp=f"{save_dir}.jpg", nrow=rows)


class Tokenizer:
    def __init__(self, dataset_const_namespace: ModuleType):
        self.constants = dataset_const_namespace
        self.rgb_transform = transforms.Compose([
            CropFace(size=WIDTH, margin=self.constants.FACE_MARGIN)
        ])

    def make_rgb_input(self, file: str, video_len: float) -> np.ndarray:
        frames = get_rgb_frames(file, video_len, ensure_frames_len=FRAMES)
        tensor_frames = [
            self.rgb_transform(
                Image.open(io.BytesIO(vid_frame))            
            ) for vid_frame in frames]
        stacked_frames = np.stack(tensor_frames, axis=0)
        if DEBUG:
            pfile = Path(file)
            save_video_frames_tensors(stacked_frames, f"{pfile.stem}")
        return stacked_frames

    def make_mfcc_input(self, file: str, sampling_rate: int) -> np.ndarray:
        audio, sr = librosa.load(file, sr=sampling_rate, duration=self.constants.MAX_AUDIO_TIME_SEC)
        assert sampling_rate == sr, f"Sampling rate from file {sr} did not match settings sampling rate {sampling_rate}"
        w_len = int(sampling_rate / 1000 * self.constants.SPEC_WINDOW_SZ_MS)
        h_len = int(sampling_rate / 1000 * self.constants.SPEC_HOP_LEN_MS)

        S = librosa.feature.melspectrogram(y=audio,
                                        fmin=0,
                                        fmax=8000,
                                        sr=sampling_rate,
                                        n_mels=128,
                                        hop_length=h_len,
                                        win_length=w_len)

        # Clamp the time dimension so it is a multiple of 10
        _, time_steps = S.shape
        time_steps = time_steps - (time_steps % 10)
        S = S[:, 0:time_steps]
        S =  librosa.power_to_db(S, ref=np.max)
        return -S

    def make_input(self, file: str, video_len: float, sampling_rate: int):
        rgb_norm, spec_norm = normalize(np.array(self.make_rgb_input(file, video_len))), normalize(self.make_mfcc_input(file, sampling_rate))
        return torch.from_numpy(rgb_norm).float(), torch.from_numpy(spec_norm).float()
    

if __name__ == '__main__':
    from datasets import enterface
    t = Tokenizer(enterface)
    t.make_rgb_input(f"{enterface.DATA_DIR}/subject 1/anger/sentence 1/s1_an_1.avi")