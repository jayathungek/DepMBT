from pathlib import Path
from typing import List, Tuple


NAME = "enterface"
MULTILABEL = False

LABEL_MAPPINGS = ["anger", "sadness", "disgust", "surprise", "happiness", "fear"]
NUM_LABELS = len(LABEL_MAPPINGS)
MAX_SPEC_SEQ_LEN = 125
SPEC_MAX_LEN = 500 
SAMPLING_RATE = 48000
SPEC_WINDOW_SZ_MS = 40
SPEC_HOP_LEN_MS   = 40
MAX_AUDIO_TIME_SEC = 5
DATA_DIR = "/root/intelpa-1/datasets/enterface database"
FACE_MARGIN = 100
FPS = 5


# first item in tuple is filepath, rest are labels
def manifest_fn(dataset_root: Path) -> List[Tuple]:
    mapping = {
        "an": 0,
        "sa": 1,
        "di": 2,
        "su": 3,
        "ha": 4,
        "fe": 5
    }
    
    mappings = []
    for p in dataset_root.rglob("*.avi"):
        label = mapping[p.stem.split("_")[1]]
        full_path = p.resolve()
        mappings.append((full_path, label))

    return mappings