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
FACE_MARGIN = 60