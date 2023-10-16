from typing import List, Tuple
from tqdm import tqdm
import pickle

import torch
from sklearn.manifold import TSNE
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

from data import load_data
from vitmbt import ViTMBT
from constants import *

CHKPT_NAME = "experiment3_110_epochs_cls-token/mbt_student_val_loss_0.00088"
PKL_PATH = f"saved_models/{CHKPT_NAME}.pkl"

def label_to_human_readable(label_tensor: torch.tensor) -> List[str]:
    assert len(label_tensor.shape) == 1, f"tensor {label_tensor} has shape {label_tensor.shape}"
    labels_readable = [LABEL_MAPPINGS[i] for i, item in 
                        enumerate(label_tensor.tolist()) if item == 1]

    if len(labels_readable) == 0:
        labels_readable = ["None"]

    return labels_readable

def load_model(path: str) -> nn.Module:
    model= ViTMBT(1024, num_class=LABELS, no_class=False, bottle_layer=BOTTLE_LAYER, freeze_first=FREEZE_FIRST, num_layers=TOTAL_LAYERS, attn_drop=ATTN_DROPOUT, linear_drop=LINEAR_DROPOUT)
    model = nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def get_embeddings_and_labels(dataloader: DataLoader, model: nn.Module) -> Tuple[np.ndarray, List[List[str]]]:
    student_outputs = []
    labels = []
    with torch.no_grad():
        for data in tqdm(dataloader):
            for video_batch, audio_batch in zip(data["student_rgb"], data["student_spec"]):
                video_batch = video_batch.to(DEVICE)
                audio_batch = audio_batch.to(DEVICE)
                student_embedding = model(audio_batch, video_batch)
                student_outputs.append(student_embedding)
                for label in data["labels"]:
                    labels.append(label_to_human_readable(label.squeeze(0)))
        
    sample_output = student_outputs[0][0]
    output_tensor = torch.FloatTensor(len(dataloader), *sample_output.shape).to(DEVICE)
    outputs = torch.cat(student_outputs, out=output_tensor)
    outputs = outputs.detach().cpu().numpy()
    assert outputs.shape[0] == len(labels), f"Have {outputs.shape[0]} embeddings and {len(labels)} labels!"
    return outputs, labels


def get_tsne_points(embeddings: np.ndarray) -> np.ndarray:
    tsne = TSNE(n_components=2)
    tsne_points = tsne.fit_transform(embeddings)
    return tsne_points


def do_inference_and_save_embeddings(dataset_path: str, model_path: str) -> Tuple[np.ndarray, List[List[str]]]:
    train_dl, _, test_dl  = load_data(dataset_path, 
                                batch_sz=BATCH_SZ,
                                train_val_test_split=[0.8, 0.1, 0.1])

    model = load_model(model_path)
    embeddings, labels = get_embeddings_and_labels(train_dl, model)
    struct = {"embeddings": embeddings, "labels": labels}
    with open(PKL_PATH, "wb") as fh:
        pickle.dump(struct, fh)


if __name__ == "__main__":
    dataset = f"{DATA_DIR}/Labels/all_pruned.csv"
    do_inference_and_save_embeddings(dataset, f"saved_models/{CHKPT_NAME}.pth")