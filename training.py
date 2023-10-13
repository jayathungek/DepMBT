import sys
from pprint import pformat
from pathlib import Path
import warnings
from argparse import ArgumentParser
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.nn import BCELoss
import torch.optim.lr_scheduler as lrsch

# from vitmbt import ViTAudio, train_audio as train, val_audio as val
# from vitmbt import ViTVideo, train_video as train, val_video as val
from vitmbt import ViTMBT, train, val
from data import load_data
from constants import *
from helpers import ClassifierMetrics
from loss import multicrop_loss


def parse_args(args):
    parser = ArgumentParser()
    parser.add_argument("script")
    parser.add_argument("-n", "--name")
    parsed = parser.parse_args(args)
    if parsed.name is None:
        print("Need model name: -n OR --name")
        exit(1)
    return parsed


parsed_args = parse_args(sys.argv)
experiment_name = parsed_args.name
save_path = Path(f"saved_models/{experiment_name}")
save_path.mkdir(exist_ok=False)
fpath_params = save_path / "hparams.txt"


mbt_teacher = ViTMBT(1024, num_class=LABELS, no_class=False, bottle_layer=BOTTLE_LAYER, freeze_first=FREEZE_FIRST, num_layers=TOTAL_LAYERS, apply_augmentation=APPLY_AUG, attn_drop=ATTN_DROPOUT, linear_drop=LINEAR_DROPOUT)
mbt_teacher = nn.DataParallel(mbt_teacher).cuda()

mbt_student = ViTMBT(1024, num_class=LABELS, no_class=False, bottle_layer=BOTTLE_LAYER, freeze_first=FREEZE_FIRST, num_layers=TOTAL_LAYERS, apply_augmentation=APPLY_AUG, attn_drop=ATTN_DROPOUT, linear_drop=LINEAR_DROPOUT)
mbt_student = nn.DataParallel(mbt_student).cuda()

train_dl, val_dl, test_dl  = load_data(f"{DATA_DIR}/Labels/all_pruned.csv", 
                                       batch_sz=BATCH_SZ,
                                       train_val_test_split=SPLIT)

# only the student's weights are updated by the optimiser
optimizer = torch.optim.AdamW(mbt_student.parameters(), betas=(0.9, 0.999), lr=LR, weight_decay=0.4)

scheduler = lrsch.SequentialLR(
    optimizer=optimizer,
    schedulers=[
    lrsch.ConstantLR(optimizer=optimizer, factor=1, total_iters=WARMUP_EPOCHS),
    lrsch.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=T_0)
], milestones=MILESTONES)


loss_func = BCELoss()
train_cls = ClassifierMetrics(task='multilabel', n_labels=LABELS, device=DEVICE)
val_cls = ClassifierMetrics(task='multilabel', n_labels=LABELS, device=DEVICE)


best = {
    "optim": optimizer.__class__.__name__,
    "optim_lr": optimizer.defaults.get('lr'),
    "optim_momentum": optimizer.defaults.get('momentum'),
    "optim_weight_decay": optimizer.defaults.get('weight_decay'),
    "loss": loss_func.__class__.__name__,
    "batch_sz": BATCH_SZ,
    "warmup_epochs": WARMUP_EPOCHS,
    "epochs": EPOCHS,
    "T_0": T_0,
    "attn_dropout": ATTN_DROPOUT,
    "pt_attn_dropout": PT_ATTN_DROPOUT,
    "linear_dropout": LINEAR_DROPOUT,
    "bottle_layer": BOTTLE_LAYER,
    "freeze_first": FREEZE_FIRST,
    "total_layers": TOTAL_LAYERS,
    "apply_augmentation": APPLY_AUG,
    "temp_teacher": TEMP_TEACHER,
    "temp_student": TEMP_STUDENT,
    "network_momentum": NETWORK_MOMENTUM,
    "centre_momentum": CENTRE_MOMENTUM,
    'num_global_views': NUM_GLOBAL_VIEWS,
    'num_local_views': NUM_LOCAL_VIEWS,
    'temp_student': TEMP_STUDENT,
    'temp_teacher': TEMP_TEACHER,
    'global_view_pct': GLOBAL_VIEW_PCT,
    'local_view_pct': LOCAL_VIEW_PCT,
    "centre_constant": CENTRE_CONSTANT,
    "best_epoch": 0,
    "val": {
        "loss": None,
    },
    "train": {
        "loss": None,
    }
}

with fpath_params.open("w") as fh:
    best_str = pformat(best)
    fh.write(best_str)
    fh.write("\n")

for epoch in range(EPOCHS):
    train_loss, updated_centre = train(mbt_teacher, mbt_student, train_dl, optimizer, CENTRE_CONSTANT, loss_fn=multicrop_loss)
    val_loss = val(mbt_teacher, mbt_student, val_dl, updated_centre, loss_fn=multicrop_loss)
    scheduler.step()

    print(
        (f"Epoch {epoch + 1}: train_loss {train_loss:.5f}, val_loss {val_loss:.5f}\n")
        )


    if best["val"]["loss"] is None or (best["val"]["loss"] is not None and val_loss < best["val"]["loss"]): 
        best["T_0"] = T_0
        best["best_epoch"] = epoch + 1
        best["val"]["loss"] = val_loss
        best["train"]["loss"] = train_loss
        fname = f"mbt_student_val_loss_{val_loss:.5f}"
        fpath_chkpt = save_path / f"{fname}.pth"
        torch.save(mbt_student.state_dict(), fpath_chkpt)


print(pformat(best))
with open(RESULTS, "a") as fh:
    best_str = pformat(best)
    fh.write(best_str)
    fh.write("\n")