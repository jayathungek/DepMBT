import os

import pandas as pd
import torch
import pickle
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from tokenizer import make_input

FRAMES = 10
CHANS = 3
WIDTH = HEIGHT = 224
SPEC_WIDTH = 800
SPEC_HEIGHT = 128
SAMPLING_RATE = 44100


class EmoDataset(Dataset):
    def __init__(self, manifest_filepath):
        super(EmoDataset, self).__init__()
        self.dataset = pd.read_csv(manifest_filepath)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset.iloc[item][0], self.dataset.iloc[item][1]


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
    spec_batch_tensor = torch.FloatTensor(len(batch), CHANS, SPEC_HEIGHT, SPEC_WIDTH)
    rgb_tensor_list = []
    spec_tensor_list = []
    labels_list = []
    last_file, last_label =  None, None
    for filename, label in batch:
        try:
            rgb, spec = make_input(filename, SAMPLING_RATE)
            rgb = rgb.reshape((CHANS * FRAMES, HEIGHT, WIDTH)).unsqueeze(0)  # f, c, h, w -> 1, c*f, h, w
            spec = spec.unsqueeze(0)                                         # c, h, w -> 1, c, h, w
            rgb_tensor_list.append(rgb)
            spec_tensor_list.append(spec)
            labels_list.append(label)
            last_file, last_label = filename, label
        except Exception as e:
            print(f"Error with {filename}: {e}")
            print(f"Using last working file again: {filename}")
            rgb, spec = make_input(last_file, SAMPLING_RATE)
            rgb = rgb.reshape((CHANS * FRAMES, HEIGHT, WIDTH)).unsqueeze(0)  # f, c, h, w -> 1, c*f, h, w
            spec = spec.unsqueeze(0)                                         # c, h, w -> 1, c, h, w
            rgb_tensor_list.append(rgb)
            spec_tensor_list.append(spec)
            labels_list.append(last_label)

    torch.cat(rgb_tensor_list, out=rgb_batch_tensor)
    torch.cat(spec_tensor_list, out=spec_batch_tensor)
    label_batch_tensor = torch.LongTensor(labels_list)
    # Don't need a mask as long as all items in the batch are guaranteed to be the same length in the time dim
    # audio_mask = torch.arange(len(batch)*SPEC_WIDTH).reshape((len(batch), SPEC_WIDTH))
    # video_mask = torch.arange(len(batch)*WIDTH).reshape((len(batch), WIDTH))
    return spec_batch_tensor.float(),rgb_batch_tensor.float(), label_batch_tensor.long()




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
    import argparse
    parser = argparse.ArgumentParser(description='Generate dataset')
    parser.add_argument('--rate', '-r', type=int, default=1, help='Downsample rate')
    parser.add_argument('--keep', '-', action='store_true', help='Keep all data in training set')
    args = parser.parse_args()
    gen_dataset(args.rate, args.keep)
