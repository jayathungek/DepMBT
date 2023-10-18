
DEVICE = "cuda"
RESULTS = "results/best.txt"
PRETRAINED_CHKPT = "./pretrained_models/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz"

FRAMES = 10
CHANS = 3
WIDTH = HEIGHT = 224
NUM_MELS = 128

JPEG_HEADER = b"\xff\xd8"
DEBUG = False

# HYPERPARAMS
NUM_GLOBAL_VIEWS = 2
NUM_LOCAL_VIEWS = 4
TEMP_STUDENT = 1
TEMP_TEACHER = 1
GLOBAL_VIEW_PCT = 0.75
LOCAL_VIEW_PCT = 0.45
NETWORK_MOMENTUM = 0.8
CENTRE_MOMENTUM = 0.8
CENTRE_CONSTANT = 0.5
WARMUP_EPOCHS = 2
EPOCHS = WARMUP_EPOCHS + 5
LR = 0.000005
WEIGHT_DECAY = 0.2
BETAS = (0.9, 0.999)
MOMENTUM = 0.9
BATCH_SZ = 8
LABELS = 8
SPLIT = [0.5, 0.5, 0.0]
MILESTONES = [WARMUP_EPOCHS]
T_0 = 6
PT_ATTN_DROPOUT = 0.15
ATTN_DROPOUT = 0.2
LINEAR_DROPOUT = 0.1
BOTTLE_LAYER = 10
FREEZE_FIRST = 8
TOTAL_LAYERS = 14
APPLY_AUG = True