from typing import List, Tuple
from tqdm import tqdm

import torch
from sklearn.manifold import TSNE
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

from data import load_data
from vitmbt import ViTMBT
from constants import *


def label_to_human_readable(label_tensor: torch.tensor) -> List[str]:
    return [LABEL_MAPPINGS[i] for i, item in 
            enumerate(label_tensor.tolist()) if item == 1]


def load_model(path: str) -> nn.Module:
    model= ViTMBT(1024, num_class=LABELS, no_class=False, bottle_layer=BOTTLE_LAYER, freeze_first=FREEZE_FIRST, num_layers=TOTAL_LAYERS, attn_drop=ATTN_DROPOUT, linear_drop=LINEAR_DROPOUT)
    model = nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def get_embeddings_and_labels(dataloader: DataLoader, model: nn.Module) -> Tuple[np.ndarray, List[List[str]]]:
    student_outputs = []
    labels = []
    for data in tqdm(dataloader):
        for video_batch, audio_batch in zip(data["student_rgb"], data["student_spec"]):
            # batch_sz = video.shape[0]
            # video = video.reshape(batch_sz * 10, 3, 224, 224)
            # video = video.reshape(batch_sz, 30, 224, 224)
            video_batch = video_batch.to(DEVICE)
            audio_batch = audio_batch.to(DEVICE)
            student_embedding = model(audio_batch, video_batch)
            student_outputs.append(student_embedding)
            for label in data["labels"]:
                labels.append(label_to_human_readable(label))
        
    sample_output = student_outputs[0][0]
    output_tensor = torch.FloatTensor(len(dataloader), *sample_output.shape)
    outputs = torch.cat(student_outputs, out=output_tensor)
    outputs = outputs.detach().cpu().numpy()
    assert outputs.shape[0] == len(labels), f"Have {outputs.shape[0]} embeddings and {len(labels)} labels!"
    return outputs, labels


def get_tsne_points(embeddings: np.ndarray) -> np.ndarray:
    # average out across emotions? and average out across bottleneck tokens?
    tsne = TSNE(n_components=2)
    avg_embeddings = np.mean(embeddings, axis=1)
    avg_embeddings = np.mean(avg_embeddings, axis=1)
    tsne_points = tsne.fit_transform(avg_embeddings)
    return tsne_points


def do_inference(dataset_path: str, model_path: str) -> Tuple[np.ndarray, List[List[str]]]:
    _, _, test_dl  = load_data(dataset_path, 
                                batch_sz=BATCH_SZ,
                                train_val_test_split=[0.8, 0.1, 0.1])

    model = load_model(model_path)
    embeddings, labels = get_embeddings_and_labels(test_dl, model)
    points2d = get_tsne_points(embeddings)
    return points2d, labels


if __name__ == "__main__":
    dataset = f"{DATA_DIR}/Labels/all_pruned.csv"
    test = np.random.rand(300, 8, 4, 1024)
    # points = get_tsne_points(test)
    # item = test[0]
    # print(mean.shape)