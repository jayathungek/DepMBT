import os

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

from tokenizer import make_input, make_input_test
from constants import *



class EmoDataset(Dataset):
    def __init__(self, manifest_filepath, nlines, sole_emotion=None):
        super(EmoDataset, self).__init__()
        self.dataset = pd.read_csv(manifest_filepath, nrows=nlines)
        self.sole_emotion = sole_emotion

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        if self.sole_emotion is None:
            num_labels = 8
            return self.dataset.iloc[item][0], [self.dataset.iloc[item][i] for i in range(1, num_labels + 1)]
        else:
            video_path = self.dataset.iloc[item][0]
            label = self.dataset.iloc[item][self.sole_emotion]
            return video_path, label



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


def collate_fn(data):
    audio, video, labels, lengths = zip(*data)
    labels = torch.tensor(labels).long()
    lengths = torch.tensor(lengths).long()
    mask = torch.arange(max(lengths))[None, :] < lengths[:, None]

    feature_audio = [torch.tensor(a).long() for a in audio]
    feature_video = [torch.tensor(v).long() for v in video]
    feature_audio = pad_sequence(feature_audio, batch_first=True, padding_value=0)
    feature_video = pad_sequence(feature_video, batch_first=True, padding_value=0)
    return feature_audio.float(), feature_video.float(), mask.long(), labels


def new_collate_fn(batch):
    rgb_batch_tensor = torch.FloatTensor(len(batch), FRAMES * CHANS, HEIGHT, WIDTH)
    spec_batch_tensor = torch.FloatTensor(len(batch), CHANS, NUM_MELS, MAX_SPEC_SEQ_LEN)
    label_batch_tensor = torch.LongTensor(len(batch), NUM_LABELS)
    rgb_tensor_list = []
    spec_tensor_list = []
    labels_list = []
    for filename, label in batch:
        rgb, spec = make_input_test(filename, SAMPLING_RATE)
        rgb = rgb.reshape((CHANS * FRAMES, HEIGHT, WIDTH)).unsqueeze(0)  # f, c, h, w -> 1, c*f, h, w
        # spec = spec.unsqueeze(0)                                         # c, h, w -> 1, c, h, w
        spec = spec.T
        label = torch.tensor([label], dtype=torch.long).unsqueeze(0)
        rgb_tensor_list.append(rgb)
        spec_tensor_list.append(spec)
        labels_list.append(label)

    torch.cat(rgb_tensor_list, out=rgb_batch_tensor)
    # torch.cat(spec_tensor_list, out=spec_batch_tensor)
    padding = (0, 0, 0, MAX_SPEC_SEQ_LEN - spec_tensor_list[0].shape[0])
    spec_tensor_list[0] = nn.ConstantPad2d(padding, 0)(spec_tensor_list[0])
    spec_batch_tensor = pad_sequence(spec_tensor_list, batch_first=True)
    spec_batch_tensor = spec_batch_tensor.swapaxes(1, 2)
    spec_batch_tensor = spec_batch_tensor.unsqueeze(1).repeat(1, 3, 1, 1)

    torch.cat(labels_list, out=label_batch_tensor)
    # label_batch_tensor = torch.vstack(labels_list)
    # label_batch_tensor = torch.LongTensor(labels_list).expand((len(batch), -1))
    # Don't need a mask as long as all items in the batch are guaranteed to be the same length in the time dim
    # audio_mask = torch.arange(len(batch)*SPEC_WIDTH).reshape((len(batch), SPEC_WIDTH))
    # video_mask = torch.arange(len(batch)*WIDTH).reshape((len(batch), WIDTH))
    return spec_batch_tensor, rgb_batch_tensor, label_batch_tensor


def load_data(data_path, batch_sz=16, train_val_test_split=[0.8, 0.1, 0.1]):
    # This is a convenience funtion that returns dataset splits of train, val and test according to the fractions specified in the arguments
    assert sum(train_val_test_split) == 1, "Train, val and test fractions should sum to 1!"  # Always a good idea to use static asserts when processing arguments that are passed in by a user!
    nlines = None
    se = None
    dataset = EmoDataset(data_path, nlines=nlines, sole_emotion=se)
    
    # This code generates the actual number of items that goes into each split using the user-supplied fractions
    tr_va_te = []
    for frac in train_val_test_split:
        actual_count = frac * len(dataset)
        actual_count = round(actual_count)
        tr_va_te.append(actual_count)
    
    # split dataset into train, val and test
    train_split, val_split, test_split = random_split(dataset, tr_va_te)
    
    # Use Pytorch DataLoader to load each split into memory. It's important to pass in our custom collate function, so it knows how to interpret the 
    # data and load it. num_workers tells the DataLoader how many CPU threads to use so that data can be loaded in parallel, which is faster
    train_dl = DataLoader(train_split, 
                          batch_size=batch_sz, 
                          shuffle=True, 
                          collate_fn=new_collate_fn)            
    val_dl = DataLoader(val_split, 
                        batch_size=batch_sz, 
                        shuffle=True, 
                        collate_fn=new_collate_fn)
    test_dl = DataLoader(test_split,
                         batch_size=batch_sz,
                         shuffle=False,
                         collate_fn=new_collate_fn)
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
    # import argparse
    # parser = argparse.ArgumentParser(description='Generate dataset')
    # parser.add_argument('--rate', '-r', type=int, default=1, help='Downsample rate')
    # parser.add_argument('--keep', '-', action='store_true', help='Keep all data in training set')
    # args = parser.parse_args()
    # gen_dataset(args.rate, args.keep)
    ds = EmoDataset("/root/intelpa-1/datasets/EmoReact/EmoReact_V_1.0/Labels/all_pruned.csv", nlines=None, sole_emotion=None)
    dl = DataLoader(ds, collate_fn=new_collate_fn, shuffle=True, batch_size=1)

    video_augmentations = nn.Sequential(
        transforms.ColorJitter(),
        transforms.GaussianBlur(),
        transforms.RandomRotation(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomInvert()
    )
    for audio, visual, label in dl:
        print(audio.shape, visual.shape, label.shape)
        plt.plot(visual)
