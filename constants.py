
DEVICE = "cuda"
PRETRAINED_CHKPT = "./pretrained_models/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz"
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