import io
import sys
import csv
from tqdm import tqdm
from pathlib import Path
from typing import List, Optional, Callable, Union

import librosa
import ffmpeg
import numpy as np
import torch
from torch import nn
import torchvision.utils
from torchvision import transforms
from PIL import Image

from timm.layers.format import Format, nchw_to
from timm.layers.helpers import to_2tuple

DEBUG = False
FRAMES = 10
CHANS = 3
WIDTH = HEIGHT = 224
SPEC_WIDTH = 800
SPEC_HEIGHT = 128
SAMPLING_RATE = 44100
DS_BASE = "/root/intelpa-2/datasets/audioset/train"
TEST_FILE = "zLo1mkKE4sw.mp4"
TEST_LABEL = "/m/02zsn,/m/09x0r"  # Female speech, woman speaking,Speech
_JPEG_HEADER = b"\xff\xd8"


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


def get_rgb_frames(video_path: str, ensure_frames_len: int = None) -> np.array:
    min_resize = 256
    new_width = "(iw/min(iw,ih))*{}".format(min_resize)
    cmd = (
        ffmpeg
        .input(video_path)
        .trim(start=0, end=10)
        .filter("fps", fps=1)
        .filter("scale", new_width, -1)
        .output("pipe:", format="image2pipe")
    )
    jpeg_bytes, _ = cmd.run(capture_stdout=True, quiet=not DEBUG)
    jpeg_bytes = jpeg_bytes.split(_JPEG_HEADER)[1:]
    jpeg_bytes = map(lambda x: _JPEG_HEADER + x, jpeg_bytes)
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


def normalize(arr: np.ndarray):
    return arr / arr.max()


def get_spectrogram(video_path: str, sampling_rate: int) -> io.BytesIO:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from librosa import display as lrd
    audio, sr = librosa.load(video_path, offset=0, duration=(10 - 0), sr=sampling_rate)
    # w_len = int(sampling_rate / 1000 * 25)  # 25ms window length
    # h_len = int(sampling_rate / 1000 * 10)  # 10ms hop length
    w_len = int(sampling_rate / 1000 * 12.5)  # 12.5ms window length
    h_len = int(sampling_rate / 1000 * 12.5)  # 12.5ms hop length

    S = librosa.feature.melspectrogram(y=audio,
                                       fmin=0,
                                       fmax=8000,
                                       sr=sampling_rate,
                                       n_mels=128,
                                       hop_length=h_len,
                                       win_length=w_len)

    # Clamp the time dimension so it is a multiple of 100
    _, time_steps = S.shape
    time_steps = time_steps - (time_steps % 100)
    S = S[:, 0:time_steps]

    fig = plt.Figure()
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    p = lrd.specshow(librosa.power_to_db(S, ref=np.max),
                     sr=sampling_rate,
                     hop_length=h_len,
                     win_length=w_len,
                     fmin=0,
                     fmax=8000,
                     ax=ax,
                     auto_aspect=False,
                     y_axis='mel',
                     x_axis='time')
    img_data = io.BytesIO()
    fig.savefig(img_data, bbox_inches='tight', pad_inches=0)
    return img_data


RGB_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.PILToTensor()
])

SPEC_TRANSFORM = transforms.Compose([
    transforms.Resize((128, 800)),
    transforms.PILToTensor()
])


def make_rgb_input(file: str) -> np.ndarray:
    frames = get_rgb_frames(file, ensure_frames_len=FRAMES)
    tensor_frames = [
        RGB_TRANSFORM(
            Image.open(io.BytesIO(vid_frame))
        ) for vid_frame in frames]
    # .permute(2, 1, 0)
    if DEBUG:
        with open("midframe.jpeg", "wb") as fh:
            fh.write(frames[5])
    return np.stack(tensor_frames, axis=0)


def make_spec_input(file: str, sampling_rate: int) -> np.ndarray:
    # output shape: 1, 1000, 128, 3
    mspec = get_spectrogram(file, sampling_rate)
    spec_tensor = SPEC_TRANSFORM(Image.open(mspec))
    spec_tensor = spec_tensor[:3, :, :] / 225.0  # remove alpha channel and normalise
    if DEBUG:
        torchvision.utils.save_image(spec_tensor, "spec_test1.bmp")
    return spec_tensor


def pos_embed(x, num_patches, embed_dim):
    # original timm, JAX, and deit vit impl
    # pos_embed has entry for class token, concat then add
    cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
    num_prefix_tokens = 1
    embed_len = num_patches + num_prefix_tokens
    pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
    x = torch.cat((cls_token.expand(x.shape[0], -1, -1), x), dim=1)
    x = x + pos_embed
    return x


def make_input(file: str, sampling_rate: int):
    rgb_norm, spec_norm = normalize(np.array(make_rgb_input(file))), normalize(
        np.array(make_spec_input(file, sampling_rate)))
    return torch.from_numpy(rgb_norm).float(), torch.from_numpy(spec_norm).float()


def prune_manifest(manifest_filepath):
    """
    runs the video files in the manifest through the pre-processing pipeline
    and excludes any failures from the final manifest
    """
    failed = 0
    ok_lines = []
    manifest_filepath = Path(manifest_filepath).resolve()
    with open(manifest_filepath, "r") as fh:
        reader = csv.reader(fh)
        for filepath, label in tqdm(reader):
            filepath = Path(filepath).resolve()
            try:
                rgb, spec = make_input(filepath, SAMPLING_RATE)
                rgb = rgb.reshape((CHANS * FRAMES, HEIGHT, WIDTH)).unsqueeze(0)  # f, c, h, w -> 1, c*f, h, w
                spec = spec.unsqueeze(0)
                ok_lines.append([str(filepath), label])
            except Exception as e:
                print(f"Failed to process {filepath.name}: {e}")
                failed += 1
    
    dest = f"{manifest_filepath.parent / manifest_filepath.stem}_pruned.csv"
    with open(dest, "w") as fh:
        writer = csv.writer(fh)
        writer.writerows(ok_lines)
        print(f"Wrote {len(ok_lines)} rows to {dest}, {failed} failed.")


    

if __name__ == '__main__':
    mfest = sys.argv[1]
    prune_manifest(mfest)
    exit()
    filenames = ["zy8mUJijw3o.mp4", "zxxjPPEujvU.mp4", "zyGjrJfE_rg.mp4", "zyXa2tdBTGc.mp4", "zye7IPXojSc.mp4"]

    rgb_batch_tensor = torch.FloatTensor(len(filenames), FRAMES*CHANS, HEIGHT, WIDTH)
    spec_batch_tensor = torch.FloatTensor(len(filenames), CHANS, SPEC_HEIGHT, SPEC_WIDTH)
    rgb_tensor_list = []
    spec_tensor_list = []
    for filename in filenames:
        rgb, spec = make_input(f"{DS_BASE}/{filename}", SAMPLING_RATE)
        rgb = rgb.reshape((CHANS * FRAMES, HEIGHT, WIDTH)).unsqueeze(0)  # f, c, h, w -> 1, c*f, h, w
        rgb_tensor_list.append(rgb)
        spec = spec.unsqueeze(0)
        spec_tensor_list.append(spec)

    torch.cat(rgb_tensor_list, out=rgb_batch_tensor)
    torch.cat(spec_tensor_list, out=spec_batch_tensor)
    print(rgb_batch_tensor.shape, spec_batch_tensor.shape)
    exit()

    patch_embed = PatchEmbed()
    patch_embed_spec = PatchEmbed(img_size=(128, 800), num_frames=1)
    rgb = patch_embed(rgb)
    spec = patch_embed_spec(spec)
    rgb = pos_embed(rgb, num_patches=patch_embed.num_patches, embed_dim=768)
    spec = pos_embed(spec, num_patches=patch_embed_spec.num_patches, embed_dim=768)
    mm_embedding = torch.cat([rgb, spec], dim=1)
    print(mm_embedding.shape)

