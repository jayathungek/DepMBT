from typing import List, Tuple
from functools import partial as P

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from helpers import is_jupyter, AverageMeter
if is_jupyter():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

from constants import *

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
    def __init__(self, in_channels, out_channels, downsample, first_temporal_kernel=1, dimensions=3):
        super().__init__()
        self.downsample = downsample
        if dimensions == 3:
            conv_module = nn.Conv3d
            bn_module = nn.BatchNorm3d
            kernel_left = (first_temporal_kernel, 1, 1)
            kernel_middle = (1, 3, 3) 
            stride_shortcut = (1, 2, 2) if downsample else 1
            stride_middle = (1, 2, 2) if downsample else 1
            if first_temporal_kernel == 3:
                padding_middle = (1, 1, 1)
            else:
                padding_middle = (0, 1, 1)
        elif dimensions == 2:
            conv_module = nn.Conv2d
            bn_module = nn.BatchNorm2d
            kernel_left = (1, 1)
            kernel_middle = (3, 3) 
            padding_middle = (1, 1)
            stride_shortcut = (2, 2) if downsample else 1
            stride_middle = (2, 2) if downsample else 1
            padding_middle = (1, 1)
        else:
            raise ValueError(f"Invalid dimension (expected 2 or 3 but got: {dimensions})")
        stride_left = 1
        kernel_shortcut = 1
        kernel_right =  1

        self.conv1 = conv_module(in_channels, 
                               out_channels//4, 
                               kernel_size=kernel_left,
                               stride=stride_left)
        self.conv2 = conv_module(out_channels//4,
                               out_channels//4,
                               kernel_size=kernel_middle,
                               stride=stride_middle,
                               padding=padding_middle)
        self.conv3 = conv_module(out_channels//4,
                               out_channels,
                               kernel_size=kernel_right,
                               stride=1)
        self.shortcut = nn.Sequential()
        
        if self.downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                conv_module(in_channels,
                          out_channels,
                          kernel_size=kernel_shortcut,
                          stride=stride_shortcut),
                bn_module(out_channels)
            )

        self.bn1 = bn_module(out_channels//4)
        self.bn2 = bn_module(out_channels//4)
        self.bn3 = bn_module(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.bn2(self.conv2(input)))
        input = nn.ReLU()(self.bn3(self.conv3(input)))
        input = input + shortcut
        return nn.ReLU()(input)

class ResNet50(nn.Module):
    def __init__(self, in_channels, resblock, repeat, dimensions=3):
        super().__init__()
        channels = [64, 256, 512, 1024, 2048]
        if dimensions == 3:
            conv_module = nn.Conv3d
            mp_module = nn.MaxPool3d
            bn_module = nn.BatchNorm3d
            ap_module = nn.AvgPool3d
            conv_kernel = (5, 7, 7)
            conv_padding = (2, 3, 3)
            mp_kernel = (1, 3, 3)
            mp_padding = (0, 1, 1)
            mp_stride = (1, 2, 2)
            ap_kernel = (2, 2, 2)
            ap_stride = (2, 2, 2)
            kernal_temp_conv45 = 3
        elif dimensions == 2:
            conv_module = nn.Conv2d
            mp_module = nn.MaxPool2d
            bn_module = nn.BatchNorm2d
            ap_module = nn.AvgPool2d
            conv_kernel = (7, 7)
            conv_padding = (3, 3)
            mp_kernel = (3, 3)
            mp_padding = (1, 1)
            mp_stride = (2, 2)
            ap_kernel = (2, 2)
            ap_stride = (2, 2)
            kernal_temp_conv45 = 1
        else:
            raise ValueError(f"dimensions arg must either be 2 or 3, got: {dimensions}")


        self.conv1 = nn.Sequential(
            conv_module(in_channels, 64, kernel_size=conv_kernel, stride=2, padding=conv_padding),
            mp_module(kernel_size=mp_kernel, stride=mp_stride, padding=mp_padding),
            bn_module(64),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential()
        self.conv2.add_module("conv2_1", P(resblock, channels[0], channels[1], downsample=False, dimensions=dimensions)())
        for i in range(1, repeat[0]):
            self.conv2.add_module(f"conv2_{i + 1}", P(resblock, channels[1], channels[1], downsample=False, dimensions=dimensions)())

        self.conv3 = nn.Sequential()
        self.conv3.add_module("conv3_1", P(resblock, channels[1], channels[2], downsample=True, dimensions=dimensions)())
        for i in range(1, repeat[1]):
            self.conv3.add_module(f"conv3_{i + 1}", P(resblock, channels[2], channels[2], downsample=False, dimensions=dimensions)())

        self.conv4 = nn.Sequential()
        self.conv4.add_module("conv4_1", P(resblock, channels[2], channels[3], first_temporal_kernel=kernal_temp_conv45, downsample=True, dimensions=dimensions)())
        for i in range(1, repeat[2]):
            self.conv4.add_module(f"conv4_{i + 1}", P(resblock, channels[3], channels[3], downsample=False, dimensions=dimensions)())

        self.conv5 = nn.Sequential()
        self.conv5.add_module("conv5_1", P(resblock, channels[3], channels[4], first_temporal_kernel=kernal_temp_conv45, downsample=True, dimensions=dimensions)())
        for i in range(1, repeat[3]):
            self.conv5.add_module(f"conv5_{i + 1}", P(resblock, channels[4], channels[4], downsample=False, dimensions=dimensions)())
        
        self.avg_pool = ap_module(kernel_size=ap_kernel, stride=ap_stride)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avg_pool(x)
        return x


class LateFusionCNN(nn.Module):
    def __init__(self, dataset_const_namespace, embed_dim=2048, hidden_sz=4096, drop=0.2, apply_augmentation=False):
        super().__init__()
        self.apply_augmentation = apply_augmentation
        self.ds_constants = dataset_const_namespace
        self.unimodal_audio = ResNet50(3, ResBlock, [3, 4, 6, 3], dimensions=2)
        self.unimodal_video = ResNet50(3, ResBlock, [3, 4, 6, 3], dimensions=3)
        self.head = nn.Sequential(
            nn.Linear(3 * 3 * (1 + 2) * embed_dim, hidden_sz), # cnn pools and downsamples result in 3x3 feature map in space, 1 time dim in audio and 2 in video 
            nn.Dropout(drop),
            nn.Linear(hidden_sz, embed_dim)
        ) 

        if apply_augmentation:
            self.video_augmentations = nn.Sequential()
        else:
            self.video_augmentations = nn.Identity()

    def forward(self, a, v):
        a = self.unimodal_audio(a)
        v = self.unimodal_video(v)
        a = a.unsqueeze(2)
        out = torch.cat((v, a), dim=2)
        out = out.flatten(start_dim=1, end_dim=4)
        out = self.head(out)
        return out

def update_teacher_net_params(t, s):
    t_copy = t.state_dict().copy()
    s_copy = s.state_dict().copy()

    for key in list(s.state_dict().keys()):
        t_copy[key] = NETWORK_MOMENTUM * t_copy[key] + (1 - NETWORK_MOMENTUM)*s_copy[key]
    return t_copy

def train_multicrop(teacher_net, student_net, trainldr, optimizer, centre, loss_fn):
    total_losses = AverageMeter()
    teacher_net.train()
    student_net.train()
    for data in tqdm(trainldr):
        # teacher outputs for each view
        teacher_outputs = []
        for video, audio in zip(data["teacher_rgb"], data["teacher_spec"]):
            batch_sz = video.shape[0]
            video = teacher_net.module.video_augmentations(video)
            video = video.to(DEVICE)
            audio = audio.to(DEVICE)
            teacher_embedding = teacher_net(audio, video)
            teacher_outputs.append(teacher_embedding)

        # student outputs for each view
        student_outputs = []
        for video, audio in zip(data["student_rgb"], data["student_spec"]):
            video = student_net.module.video_augmentations(video)
            video = video.to(DEVICE)
            audio = audio.to(DEVICE)
            student_embedding = student_net(audio, video)
            student_outputs.append(student_embedding)

        optimizer.zero_grad()
        loss = loss_fn(teacher_outputs, student_outputs, centre)
        loss.backward()
        optimizer.step()
        new_t_params = update_teacher_net_params(teacher_net, student_net) 
        teacher_net.load_state_dict(new_t_params)
        centre = CENTRE_MOMENTUM * centre + (1 - CENTRE_MOMENTUM) * torch.cat(teacher_outputs).detach().mean()
        total_losses.update(loss.data.item(), batch_sz)

    return total_losses.avg(), centre
        

if __name__ == "__main__":
    from datasets import enterface

    audio = torch.rand(8, 3, 224, 224)
    video = torch.rand(8, 3, 16, 224, 224)
    # r3d = ResNet50(3, ResBlock, [3, 4, 6, 3], dimensions=2)
    # out = r3d(audio)
    # print(out.shape)
    # exit()
    lfc = LateFusionCNN(enterface)
    out = lfc(audio, video)
    print(out.shape)
    