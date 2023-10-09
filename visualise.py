from typing import List

import torch
from data import load_data


LABEL_MAPPINGS = ["Curiosity", "Uncertainty", "Excitement", "Happiness", "Surprise", "Disgust", "Fear", "Frustration"]

def label_to_human_readable(label_tensor: torch.tensor) -> List[str]:
    return [LABEL_MAPPINGS[i] for i, item in 
            enumerate(label_tensor.tolist()) if item == 1]

def load_model(path):
    pass

def get_tsne_points():
    pass



if __name__ == "__main__":
    test = torch.rand((300, 8, 4, 1024))