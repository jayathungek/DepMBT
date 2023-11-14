from typing import List, Tuple
from types import ModuleType
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


# TODO: Implement Calinski-Harabasz Index algorithm to check the clustering performance 

def get_tsne_points(embeddings: np.ndarray) -> np.ndarray:
    tsne = TSNE(n_components=2, perplexity=50)
    tsne_points = tsne.fit_transform(embeddings)
    return tsne_points


class Visualiser:
    def __init__(self, dataset_const_namespace: ModuleType) -> None:
        self.ds_constants = dataset_const_namespace

    def label_to_human_readable(self, label_tensor: torch.tensor) -> List[str]:
        assert len(label_tensor.shape) == 1, f"tensor {label_tensor} has shape {label_tensor.shape}"
        if self.ds_constants.MULTILABEL:
            labels_readable = [self.ds_constants.LABEL_MAPPINGS[i] for i, item in 
                                enumerate(label_tensor.tolist()) if item == 1]
        else:
            labels_readable = [self.ds_constants.LABEL_MAPPINGS[i] for i in label_tensor.tolist()]

        if len(labels_readable) == 0:
            labels_readable = ["None"]

        return labels_readable

    def load_model(self, path: str) -> (nn.Module, float):
        model= ViTMBT(self.ds_constants, 1024, num_class=LABELS, no_class=False, bottle_layer=BOTTLE_LAYER, freeze_first=FREEZE_FIRST, num_layers=TOTAL_LAYERS, attn_drop=ATTN_DROPOUT, linear_drop=LINEAR_DROPOUT)
        model = nn.DataParallel(model).cuda()
        save_items = torch.load(path)
        model.load_state_dict(save_items["student_state_dict"])
        model.eval()
        return model, save_items["centre"]


    def get_embeddings_and_labels(self, dataloader: DataLoader, model: nn.Module) -> Tuple[np.ndarray, List[List[str]]]:
        student_outputs = []
        labels = []
        with torch.no_grad():
            for data in tqdm(dataloader):
                # we only use the first item in the student crops, which is a global view of the 
                # scene
                video_batch, audio_batch = data["student_rgb"][0], data["student_spec"][0]
                labels_batch = data["labels"]
                video_batch = video_batch.to(DEVICE)
                audio_batch = audio_batch.to(DEVICE)
                student_embedding = model(audio_batch, video_batch)
                student_outputs.append(student_embedding)
                for label in labels_batch:
                    flattened_labels = label.flatten()
                    labels.append(self.label_to_human_readable(flattened_labels))
            
        sample_output = student_outputs[0][0]
        output_tensor = torch.FloatTensor(len(dataloader), *sample_output.shape).to(DEVICE)
        outputs = torch.cat(student_outputs, out=output_tensor)
        outputs = outputs.detach().cpu().numpy()
        assert outputs.shape[0] == len(labels), f"Have {outputs.shape[0]} embeddings and {len(labels)} labels!"
        return outputs, labels


    def do_inference_and_save_embeddings(self, model_path: str, split_seed: int=None) -> Tuple[np.ndarray, List[List[str]]]:
        train_dl, _, test_dl, _ = load_data(self.ds_constants, 
                                            batch_sz=BATCH_SZ,
                                            train_val_test_split=[0.5, 0.5, 0],
                                            seed=split_seed)

        model, _ = self.load_model(model_path)
        embeddings, labels = self.get_embeddings_and_labels(train_dl, model)
        struct = {"embeddings": embeddings, "labels": labels}
        with open(PKL_PATH, "wb") as fh:
            pickle.dump(struct, fh)


if __name__ == "__main__":
    from datasets import emoreact, enterface

    CHKPT_NAME = "enterface_mbt/mbt_student_train_loss_0.01266"
    PKL_PATH = f"saved_models/{CHKPT_NAME}.pkl"
    v = Visualiser(enterface)
    v.do_inference_and_save_embeddings(f"saved_models/{CHKPT_NAME}.pth")