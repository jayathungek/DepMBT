from pathlib import Path
from typing import List, Tuple
import pandas as pd

NAME = "emoreact"
MULTILABEL = True

LABEL_MAPPINGS = ["Curiosity", "Uncertainty", "Excitement", "Happiness", "Surprise", "Disgust", "Fear", "Frustration"]
NUM_LABELS = len(LABEL_MAPPINGS)
MAX_SPEC_SEQ_LEN = 375
SPEC_MAX_LEN = 1000 
SAMPLING_RATE = 44100
SPEC_WINDOW_SZ_MS = 40
SPEC_HOP_LEN_MS   = 40
MAX_AUDIO_TIME_SEC = 15
DATA_DIR = "/root/intelpa-1/datasets/EmoReact/EmoReact_V_1.0"
NAMES_FILE = "Labels/all_names.txt"
LABELS_FILE = "Labels/all_labels.txt"
FACE_MARGIN = 60
FPS = 1
DATA_EXT = "mp4"


def append_cols_right(names: Path, 
                      names_header: List[str], 
                      labels: Path, 
                      labels_header: List[str]) -> pd.DataFrame:
    """
    Appends the columns of file2 to the columns of file1
    """
    data_dir = Path(DATA_DIR) / "Data"
    names_df = pd.read_csv(names, names=names_header)
    labels_df = pd.read_csv(labels, names=labels_header)
    labels_df.reset_index() # label values are not unique!
    assert names_df.shape[1] == len(names_header), "Mismatch of column count and header name count in file 1"
    assert labels_df.shape[1] == len(labels_header), "Mismatch of column count and header name count in file 2"
    assert names_df.shape[0] == labels_df.shape[0], "Mismatch of number of rows between file 1 and 2"
    joined = pd.concat([names_df, labels_df], axis=1)
    joined['filename'] = joined['filename'].map(lambda s: str(data_dir / s.replace("'", "")))
    return joined


def manifest_fn(dataset_root: Path) -> List[List]:
    names = dataset_root / NAMES_FILE
    labels = dataset_root / LABELS_FILE
    manifest_df = append_cols_right(names, ["filename"], labels, LABEL_MAPPINGS)
    mappings = [
        manifest_df.loc[i, :].values.flatten().tolist()
        for i in range(len(manifest_df))
    ]
    return mappings

