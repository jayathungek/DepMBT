
DEVICE = "cuda"
RESULTS = "results/best.txt"
PRETRAINED_CHKPT = "./pretrained_models/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz"
LABEL_MAPPINGS = ["Curiosity", "Uncertainty", "Excitement", "Happiness", "Surprise", "Disgust", "Fear", "Frustration"]

FRAMES = 10
CHANS = 3
WIDTH = HEIGHT = 224
NUM_LABELS = 8
NUM_MELS = 128
MAX_SPEC_SEQ_LEN = 375
SPEC_MAX_LEN = 1000 
SAMPLING_RATE = 44100
SPEC_WINDOW_SZ_MS = 40
SPEC_HOP_LEN_MS   = 40
MAX_AUDIO_TIME_SEC = 15
DATA_DIR = "/root/intelpa-1/datasets/EmoReact/EmoReact_V_1.0"


FACE_MARGIN = 60
SAMPLING_RATE_AS = 22050
DS_BASE = "/root/intelpa-2/datasets/audioset/train"
TEST_FILE = "zLo1mkKE4sw.mp4"
TEST_LABEL = "/m/02zsn,/m/09x0r"  # Female speech, woman speaking,Speech
JPEG_HEADER = b"\xff\xd8"
DEBUG = False

# HYPERPARAMS
NUM_GLOBAL_VIEWS = 2
NUM_LOCAL_VIEWS = 3
TEMP_STUDENT = 0.06
TEMP_TEACHER = 0.06
GLOBAL_VIEW_PCT = 0.7
LOCAL_VIEW_PCT = 0.25
NETWORK_MOMENTUM = 0.9
CENTRE_MOMENTUM = 0.9
CENTRE_CONSTANT = 1

WARMUP_EPOCHS = 5
EPOCHS = WARMUP_EPOCHS + 30
LR = 0.00005
BETAS = (0.9, 0.999)
MOMENTUM = 0.9
BATCH_SZ = 8
LABELS = 8
SPLIT = [0.95, 0.05, 0.0]
MILESTONES = [WARMUP_EPOCHS]
T_0 = 6
PT_ATTN_DROPOUT = 0.15
ATTN_DROPOUT = 0.4
LINEAR_DROPOUT = 0.1
BOTTLE_LAYER = 10
FREEZE_FIRST = 8
TOTAL_LAYERS = 14
APPLY_AUG = True