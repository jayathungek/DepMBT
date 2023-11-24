import os
import sys
import random
from types import ModuleType
from pathlib import Path

import pandas as pd
import torch
import pickle
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from torchvision import transforms as vtrans
from torchaudio import transforms as atrans


from tokenizer import Tokenizer, get_clip_start_frames
from multicrop import MultiCrop
from constants import *


#TODO
video_augmentations = nn.Sequential(
    #crop -> area % ~ [0.3, 1], aspect ratio ~ [1:2, 2:1]
    #resize -> 224, 224
    #flip -> {0, 1} uniform
    #jitter -> {0, 1} 1 weighted 80%
    #grayscale -> {0, 1} 1 weighted 20%
    #gaussian blur
)


class EmoDataset(Dataset):
    def __init__(self, dataset_const_namespace, nlines, sole_emotion=None):
        super(EmoDataset, self).__init__()
        self.constants = dataset_const_namespace
        manifest_filepath = Path(dataset_const_namespace.DATA_DIR) / f"{dataset_const_namespace.NAME}_pruned.csv"
        self.dataset = pd.read_csv(manifest_filepath, nrows=nlines)
        self.sole_emotion = sole_emotion

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        if self.constants.MULTILABEL:
            if self.sole_emotion is None:
                return (self.dataset.iloc[item][0], 
                       self.dataset.iloc[item][1], 
                       self.dataset.iloc[item][2], 
                       [self.dataset.iloc[item][i] for i in range(3, self.constants.NUM_LABELS + 3)]
                )
            else:
                video_path = self.dataset.iloc[item][0]
                video_len = self.dataset.iloc[item][1]
                video_frames = self.dataset.iloc[item][2]
                label = self.dataset.iloc[item][3 + self.sole_emotion]
                return video_path, video_len, video_frames, label
        else:
            return (
                self.dataset.iloc[item][0],
                self.dataset.iloc[item][1],
                self.dataset.iloc[item][2],
                self.dataset.iloc[item][3]
            )


class Collate_Multicrop:
    def __init__(self, dataset_namespace: ModuleType, force_audio_aspect=False):
        """
        force_audio_aspect: force resize of audio spectrogram to video shape
        """
        self.dataset_constants = dataset_namespace
        self.force_audio_shape = force_audio_aspect
        self.tokenizer = Tokenizer(dataset_namespace)
        self.audio_transform = vtrans.Resize((HEIGHT, WIDTH))
        self.multicrop_rgb = MultiCrop(image_size=(HEIGHT, WIDTH), num_global_views=NUM_GLOBAL_VIEWS, num_local_views=NUM_LOCAL_VIEWS, global_view_pct=GLOBAL_VIEW_PCT, local_view_pct=LOCAL_VIEW_PCT)
        image_sz = (HEIGHT, WIDTH) if self.force_audio_shape else (NUM_MELS, self.dataset_constants.MAX_SPEC_SEQ_LEN)
        self.multicrop_spec = MultiCrop(image_size=image_sz, num_global_views=NUM_GLOBAL_VIEWS, num_local_views=NUM_LOCAL_VIEWS, global_view_pct=GLOBAL_VIEW_PCT, local_view_pct=LOCAL_VIEW_PCT)

    def __call__(self, batch):
        
        if self.force_audio_shape:
            spec_batch_tensor = torch.FloatTensor(len(batch), CHANS, HEIGHT, WIDTH)
        else:
            spec_batch_tensor = torch.FloatTensor(len(batch), CHANS, NUM_MELS, self.dataset_constants.MAX_SPEC_SEQ_LEN)
        rgb_batch_tensor = torch.FloatTensor(len(batch), FRAMES, CHANS, HEIGHT, WIDTH)
        label_batch_tensor = torch.LongTensor(len(batch), self.dataset_constants.NUM_LABELS)
        rgb_tensor_list = []
        spec_tensor_list = []
        label_list = []
        for filename, duration, num_frames, label in batch:
            rgb, spec = self.tokenizer.make_input(filename, float(duration), self.dataset_constants.SAMPLING_RATE)
            rgb = rgb.reshape((CHANS, FRAMES, HEIGHT, WIDTH)).unsqueeze(0)  # f, c, h, w -> 1, c*f, h, w
            label = torch.tensor([label], dtype=torch.long).unsqueeze(0)
            spec = spec.T
            rgb_tensor_list.append(rgb)
            spec_tensor_list.append(spec)
            label_list.append(label)


        torch.cat(label_list, out=label_batch_tensor)
        torch.cat(rgb_tensor_list, out=rgb_batch_tensor)
        if not self.force_audio_shape:
            padding = (0, 0, 0, self.dataset_constants.MAX_SPEC_SEQ_LEN - spec_tensor_list[0].shape[0])
            spec_tensor_list[0] = nn.ConstantPad2d(padding, 0)(spec_tensor_list[0])
            spec_batch_tensor = pad_sequence(spec_tensor_list, batch_first=True)
            spec_batch_tensor = spec_batch_tensor.swapaxes(1, 2)
        else:
            torch.cat(
                [self.audio_transform(s.unsqueeze(0)) for s in spec_tensor_list],
                dim=0,
                out=spec_batch_tensor
            )
        spec_batch_tensor = spec_batch_tensor.unsqueeze(1).repeat(1, 3, 1, 1)
        rgb_teacher_views, rgb_student_views = self.multicrop_rgb(rgb_batch_tensor)
        spec_teacher_views, spec_student_views = self.multicrop_spec(spec_batch_tensor)

        return {
            "teacher_rgb": rgb_teacher_views,
            "student_rgb": rgb_student_views,
            "teacher_spec": spec_teacher_views,
            "student_spec": spec_student_views,
            "labels": label_batch_tensor
        } 


def frame_to_audio_sample(frame: int, fps: int, audio_sr: int) -> int:
    timestamp = frame * (1 / fps)
    sample = int(audio_sr * timestamp)
    return sample


class Collate_Constrastive:
    def __init__(self, dataset_namespace: ModuleType, force_audio_aspect=False):
        """
        force_audio_aspect: force resize of audio spectrogram to video shape
        """
        self.dataset_constants = dataset_namespace
        self.force_audio_shape = force_audio_aspect
        self.tokenizer = Tokenizer(dataset_namespace)
        self.audio_transform = nn.Sequential(
            atrans.MelSpectrogram(
                sample_rate=self.dataset_constants.SAMPLING_RATE,
                n_fft=1024,
                n_mels=128,
                win_length=None,
                hop_length=512,
            ),
            atrans.AmplitudeToDB(),
            vtrans.Resize((HEIGHT, WIDTH)) if self.force_audio_shape else nn.Identity()
        )

    def __call__(self, batch):
        
        if self.force_audio_shape:
            spec_batch_tensor1 = torch.FloatTensor(len(batch), CHANS, HEIGHT, WIDTH)
            spec_batch_tensor2 = torch.FloatTensor(len(batch), CHANS, HEIGHT, WIDTH)
        else:
            spec_batch_tensor1 = torch.FloatTensor(len(batch), CHANS, NUM_MELS, self.dataset_constants.MAX_SPEC_SEQ_LEN)
            spec_batch_tensor2 = torch.FloatTensor(len(batch), CHANS, NUM_MELS, self.dataset_constants.MAX_SPEC_SEQ_LEN)
        rgb_batch_tensor1 = torch.FloatTensor(len(batch), FRAMES, CHANS, HEIGHT, WIDTH)
        rgb_batch_tensor2 = torch.FloatTensor(len(batch), FRAMES, CHANS, HEIGHT, WIDTH)
        label_batch_tensor = torch.LongTensor(len(batch), self.dataset_constants.NUM_LABELS)
        frames_count_batch_tensor = torch.LongTensor(len(batch))
        rgb_tensor_list1 = []
        rgb_tensor_list2 = []
        spec_tensor_list1 = []
        spec_tensor_list2 = []
        label_list = []
        for filename, duration, frames_count, label in batch:
            start1, start2 = get_clip_start_frames(frames_count, FRAMES)
            end1, end2 = start1 + FRAMES, start2 + FRAMES
            rgb_frames, audio_samples = self.tokenizer.make_input(filename, float(duration), self.dataset_constants.SAMPLING_RATE, constant_fps=True)
            rgb1 = rgb_frames[start1 : end1]
            rgb2 = rgb_frames[start2 : end2]

            sample_start1 = frame_to_audio_sample(start1, self.dataset_constants.VIDEO_FPS, self.dataset_constants.SAMPLING_RATE)
            sample_start2 = frame_to_audio_sample(start2, self.dataset_constants.VIDEO_FPS, self.dataset_constants.SAMPLING_RATE)
            sample_end1 = frame_to_audio_sample(end1, self.dataset_constants.VIDEO_FPS, self.dataset_constants.SAMPLING_RATE)
            sample_end2 = frame_to_audio_sample(end2, self.dataset_constants.VIDEO_FPS, self.dataset_constants.SAMPLING_RATE)
            audio1 = audio_samples[sample_start1: sample_end1]
            audio2 = audio_samples[sample_start2: sample_end2]

            spec1 = self.audio_transform(audio1.unsqueeze(0)).unsqueeze(0).repeat(1, 3, 1, 1)
            spec1_tmp = spec1 + abs(spec1.min())
            spec1_norm = spec1_tmp / spec1_tmp.max()
            spec2 = self.audio_transform(audio2.unsqueeze(0)).unsqueeze(0).repeat(1, 3, 1, 1)
            spec2_tmp = spec2 + abs(spec2.min())
            spec2_norm = spec2_tmp / spec2_tmp.max()


            label = torch.tensor([label], dtype=torch.long).unsqueeze(0)
            rgb_tensor_list1.append(rgb1.unsqueeze(0))
            rgb_tensor_list2.append(rgb2.unsqueeze(0))
            spec_tensor_list1.append(spec1_norm)
            spec_tensor_list2.append(spec2_norm)
            label_list.append(label)


        torch.cat(label_list, out=label_batch_tensor)
        torch.cat(rgb_tensor_list1, out=rgb_batch_tensor1)
        torch.cat(rgb_tensor_list2, out=rgb_batch_tensor2)
        torch.cat(spec_tensor_list1, out=spec_batch_tensor1)
        torch.cat(spec_tensor_list2, out=spec_batch_tensor2)
        if not self.force_audio_shape:
            padding = (0, 0, 0, self.dataset_constants.MAX_SPEC_SEQ_LEN - spec_tensor_list1[0].shape[0])
            padding = (0, 0, 0, self.dataset_constants.MAX_SPEC_SEQ_LEN - spec_tensor_list1[0].shape[0])
            spec_tensor_list1[0] = nn.ConstantPad2d(padding, 0)(spec_tensor_list1[0])
            spec_tensor_list2[0] = nn.ConstantPad2d(padding, 0)(spec_tensor_list2[0])
            spec_batch_tensor1 = pad_sequence(spec_tensor_list1, batch_first=True)
            spec_batch_tensor2 = pad_sequence(spec_tensor_list2, batch_first=True)
            spec_batch_tensor1 = spec_batch_tensor1.swapaxes(1, 2)
            spec_batch_tensor2 = spec_batch_tensor2.swapaxes(1, 2)
            spec_batch_tensor1 = spec_batch_tensor1.unsqueeze(1).repeat(1, 3, 1, 1)
            spec_batch_tensor2 = spec_batch_tensor2.unsqueeze(1).repeat(1, 3, 1, 1)

        return {
            "clip0": (rgb_batch_tensor1, spec_batch_tensor1),
            "clip1": (rgb_batch_tensor2, spec_batch_tensor2),
            "labels": label_batch_tensor
        } 

def load_data(dataset_const_namespace, collate_func, batch_sz=16, train_val_test_split=[0.8, 0.1, 0.1], nlines=None, se=None, seed=None, force_audio_aspect=False):
    # This is a convenience funtion that returns dataset splits of train, val and test according to the fractions specified in the arguments
    assert sum(train_val_test_split) == 1, "Train, val and test fractions should sum to 1!"  # Always a good idea to use static asserts when processing arguments that are passed in by a user!
    dataset = EmoDataset(dataset_const_namespace, nlines=nlines, sole_emotion=se)
    
    # This code generates the actual number of items that goes into each split using the user-supplied fractions
    tr_va_te = []
    for frac in train_val_test_split:
        actual_count = frac * len(dataset)
        actual_count = round(actual_count)
        tr_va_te.append(actual_count)
    
    # if there is a mismatch, make up the difference by adding it to train samples
    ds_sum = len(dataset)
    split_sum = sum(tr_va_te)
    if ds_sum != split_sum:
        diff = max(ds_sum, split_sum) - min(ds_sum, split_sum)
        tr_va_te[0] += diff

    if seed:
        split_seed = seed
    else:
        split_seed = random.randint(0, sys.maxsize)
    generator = torch.Generator().manual_seed(split_seed)
    train_split, val_split, test_split = random_split(dataset, tr_va_te, generator=generator)
    
    # Use Pytorch DataLoader to load each split into memory. It's important to pass in our custom collate function, so it knows how to interpret the 
    # data and load it. num_workers tells the DataLoader how many CPU threads to use so that data can be loaded in parallel, which is faster
    if len(train_split) > 0:
        train_dl = DataLoader(train_split, 
                            batch_size=batch_sz, 
                            shuffle=True, 
                            collate_fn=collate_func)            
    else:
        train_dl = None

    if len(val_split) > 0:
        val_dl = DataLoader(val_split, 
                            batch_size=batch_sz, 
                            shuffle=True, 
                            collate_fn=collate_func)
    else:
        val_dl = None

    if len(test_split) > 0:
        test_dl = DataLoader(test_split,
                            batch_size=batch_sz,
                            shuffle=False,
                            collate_fn=collate_func)
    else:
        test_dl = None

    return train_dl, val_dl, test_dl, split_seed



def gen_dataset(rate, keep):
    data_dir = '/root/intelpa-2/datasets'
    feat_dir = os.path.join(data_dir, 'dvlog-dataset')
    label_file = os.path.join(feat_dir, 'labels.csv')
    label_index = {"depression": 1, "normal": 0}

    dataset ={"train": [], "test": [], "valid": []}

    with open(label_file, 'r', encoding='utf-8') as f:
        data_file = f.readlines()[1:]

    for i, data in tqdm(enumerate(data_file)):
        index, label, duration, gender, fold = data.strip().split(',')

        audio = np.load(os.path.join(feat_dir, index, index+'_acoustic.npy'))
        visual = np.load(os.path.join(feat_dir, index, index+'_visual.npy'))

        leng = min(audio.shape[0], visual.shape[0])

        audio = audio[:leng, :]
        visual = visual[:leng, :]
        if fold == 'train' and keep:
            for j in range(rate):
                a = audio[j::rate, :]
                v = visual[j::rate, :]
                dataset[fold].append((a, v, label_index[label]))
        else:
            a = audio[::rate, :]
            v = visual[::rate, :]
            dataset[fold].append((a, v, label_index[label]))

    for fold in dataset.keys():
        k = 'k' if keep else ''
        with open(os.path.join(data_dir, '{}_{}{}.pickle'.format(fold, k, str(rate))), 'wb') as handle:
            pickle.dump(dataset[fold], handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__=="__main__":
    from datasets import enterface
    from tokenizer import save_video_frames_tensors

    BATCH_SZ = 3
    SPLIT = [1, 0.0, 0.0]
    dataset_to_use = enterface
    train_dl, val_dl, test_dl, ss  = load_data(dataset_to_use, 
                                        batch_sz=BATCH_SZ,
                                        train_val_test_split=SPLIT,
                                        seed=8294641842899686597)
    
    print(f"Seed: {ss}")
    # for data in train_dl:
    #     print(data["teacher_spec"][0][0][0][35])