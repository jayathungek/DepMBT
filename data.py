import os
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
from torchvision import transforms
import matplotlib.pyplot as plt

from tokenizer import Tokenizer
from multicrop import MultiCrop
from constants import *


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
                       [self.dataset.iloc[item][i] for i in range(1, self.constants.NUM_LABELS + 2)]
                )
            else:
                video_path = self.dataset.iloc[item][0]
                video_len = self.dataset.iloc[item][1]
                label = self.dataset.iloc[item][1 + self.sole_emotion]
                return video_path, video_len, label
        else:
            return self.dataset.iloc[item][0], self.dataset.iloc[item][1], self.dataset.iloc[item][2]



class DVlog(Dataset):
    def __init__(self, filename):
        super(DVlog, self).__init__()

        with open(filename, 'rb') as handle:
            self.dataset = pickle.load(handle)

        self.length = [d[0].shape[0] for d in self.dataset]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx][0], self.dataset[idx][1], self.dataset[idx][2], self.length[idx]

class Collate_fn:
    def __init__(self, dataset_namespace: ModuleType):
        self.dataset_constants = dataset_namespace
        self.multicrop_rgb = MultiCrop(image_size=(HEIGHT, WIDTH), num_global_views=NUM_GLOBAL_VIEWS, num_local_views=NUM_LOCAL_VIEWS, global_view_pct=GLOBAL_VIEW_PCT, local_view_pct=LOCAL_VIEW_PCT)
        self.multicrop_spec = MultiCrop(image_size=(NUM_MELS, self.dataset_constants.MAX_SPEC_SEQ_LEN), num_global_views=NUM_GLOBAL_VIEWS, num_local_views=NUM_LOCAL_VIEWS, global_view_pct=GLOBAL_VIEW_PCT, local_view_pct=LOCAL_VIEW_PCT)
        self.tokenizer = Tokenizer(dataset_namespace)

    def __call__(self, batch):
        rgb_batch_tensor = torch.FloatTensor(len(batch), FRAMES * CHANS, HEIGHT, WIDTH)
        spec_batch_tensor = torch.FloatTensor(len(batch), CHANS, NUM_MELS, self.dataset_constants.MAX_SPEC_SEQ_LEN)
        label_batch_tensor = torch.LongTensor(len(batch), self.dataset_constants.NUM_LABELS)
        rgb_tensor_list = []
        spec_tensor_list = []
        label_list = []
        for filename, duration, label in batch:
            rgb, spec = self.tokenizer.make_input(filename, float(duration), self.dataset_constants.SAMPLING_RATE)
            rgb = rgb.reshape((CHANS * FRAMES, HEIGHT, WIDTH)).unsqueeze(0)  # f, c, h, w -> 1, c*f, h, w
            label = torch.tensor([label], dtype=torch.long).unsqueeze(0)
            spec = spec.T
            rgb_tensor_list.append(rgb)
            spec_tensor_list.append(spec)
            label_list.append(label)


        torch.cat(label_list, out=label_batch_tensor)
        torch.cat(rgb_tensor_list, out=rgb_batch_tensor)
        padding = (0, 0, 0, self.dataset_constants.MAX_SPEC_SEQ_LEN - spec_tensor_list[0].shape[0])
        spec_tensor_list[0] = nn.ConstantPad2d(padding, 0)(spec_tensor_list[0])
        spec_batch_tensor = pad_sequence(spec_tensor_list, batch_first=True)
        spec_batch_tensor = spec_batch_tensor.swapaxes(1, 2)
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


def load_data(dataset_const_namespace, batch_sz=16, train_val_test_split=[0.8, 0.1, 0.1], nlines=None, se=None):
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
    train_split, val_split, test_split = random_split(dataset, tr_va_te)
    
    # Use Pytorch DataLoader to load each split into memory. It's important to pass in our custom collate function, so it knows how to interpret the 
    # data and load it. num_workers tells the DataLoader how many CPU threads to use so that data can be loaded in parallel, which is faster
    new_collate_fn = Collate_fn(dataset_const_namespace)
    if len(train_split) > 0:
        train_dl = DataLoader(train_split, 
                            batch_size=batch_sz, 
                            shuffle=True, 
                            collate_fn=new_collate_fn)            
    else:
        train_dl = None

    if len(val_split) > 0:
        val_dl = DataLoader(val_split, 
                            batch_size=batch_sz, 
                            shuffle=True, 
                            collate_fn=new_collate_fn)
    else:
        val_dl = None

    if len(test_split) > 0:
        test_dl = DataLoader(test_split,
                            batch_size=batch_sz,
                            shuffle=False,
                            collate_fn=new_collate_fn)
    else:
        test_dl = None

    return train_dl, val_dl, test_dl



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
    train_dl, val_dl, test_dl  = load_data(dataset_to_use, 
                                        batch_sz=BATCH_SZ,
                                        train_val_test_split=SPLIT)
    for data in train_dl:
        pass