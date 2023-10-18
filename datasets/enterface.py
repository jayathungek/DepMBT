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
FACE_MARGIN = 60