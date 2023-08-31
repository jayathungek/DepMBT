import torch
from torch.utils.data import DataLoader
from torch.nn import BCELoss
import torch.nn as nn
from pprint import pformat

from importlib import reload

import vitmbt
import mbt
import ablation
import helpers
reload(mbt)
reload(ablation)
reload(helpers)
reload(vitmbt)
from helpers import ClassifierMetrics, display_tensor_as_rgb
from vitmbt import ViTMBT, train, val
# from vitunimodal import ViTMBT, train, val
from data import EmoDataset, new_collate_fn, load_data
import warnings
warnings.filterwarnings('ignore')

from constants import *



DEVICE = "cuda"
RESULTS = "results/best.txt"
PRETRAINED_CHKPT = "./pretrained_models/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz"
EPOCHS = 15
lr = 0.00001
betas = (0.9, 0.999)
momentum = 0.9
BATCH_SZ = 16
LABELS = 8
SPLIT = [0.8, 0.1, 0.1]

vmbt = ViTMBT(1024, num_class=LABELS, no_class=False)
vmbt = nn.DataParallel(vmbt).cuda()


train_dl, val_dl, test_dl  = load_data(f"{DATA_DIR}/Labels/all_pruned.csv", 
                                       batch_sz=BATCH_SZ,
                                       train_val_test_split=SPLIT)

optimizer = torch.optim.AdamW(vmbt.parameters(), betas=(0.9, 0.999), lr=lr, weight_decay=1.0 / BATCH_SZ)
# optimizer = torch.optim.SGD(vmbt.parameters(), lr=lr, momentum=momentum, weight_decay=1/BATCH_SZ)

loss_func = BCELoss()
train_cls = ClassifierMetrics(task='multilabel', n_labels=LABELS, device=DEVICE)
val_cls = ClassifierMetrics(task='multilabel', n_labels=LABELS, device=DEVICE)
# loss_func = nn.CrossEntropyLoss()

best = {
    "optim": optimizer.__class__.__name__,
    "optim_lr": optimizer.defaults.get('lr'),
    "optim_momentum": optimizer.defaults.get('momentum'),
    "optim_weight_decay": optimizer.defaults.get('weight_decay'),
    "loss": loss_func.__class__.__name__,
    "batch_sz": BATCH_SZ,
    "epochs": EPOCHS,
    "val": {
        "loss": None,
        "f1": None,
        "recall": None,
        "precision": None,
        "acc": None
    },
    "train": {
        "loss": None,
        "f1": None,
        "recall": None,
        "precision": None,
        "acc": None
    }
}

for epoch in range(EPOCHS):
    train_loss, train_metrics = train(vmbt, train_dl, optimizer, loss_fn=loss_func, cls_metrics=train_cls)
    val_loss, val_metrics = val(vmbt, val_dl, loss_fn=loss_func, cls_metrics=val_cls)

    val_loss_val = val_loss
    val_f1_val = val_metrics.f1.item()
    val_recall_val = val_metrics.recall.item()
    val_precision_val = val_metrics.precision.item()
    val_acc_val = val_metrics.acc.item()

    train_loss_val = train_loss
    train_f1_val = train_metrics.f1.item()
    train_recall_val = train_metrics.recall.item()
    train_precision_val = train_metrics.precision.item()
    train_acc_val = train_metrics.acc.item()

    print(
        (f"Epoch {epoch + 1}: train_loss {train_loss_val:.5f}, val_loss {val_loss_val:.5f}\n"
            f"                   train_precision {train_precision_val}, val_precision {val_precision_val}\n"
            f"                   train_recall {train_recall_val}, val_recall {val_recall_val}\n"
            f"                   train_f1 {train_f1_val}, val_f1 {val_f1_val}"
            )
        )

    if best["val"]["f1"] is None or (best["val"]["f1"] is not None and val_f1_val > best["val"]["f1"]): 
        best["val"]["loss"] = val_loss
        best["val"]["f1"] = val_f1_val
        best["val"]["recall"] = val_recall_val
        best["val"]["precision"] = val_precision_val
        best["val"]["acc"] = val_acc_val
        best["train"]["loss"] = train_loss
        best["train"]["f1"] = train_f1_val
        best["train"]["recall"] = train_recall_val
        best["train"]["precision"] = train_precision_val
        best["train"]["acc"] = train_acc_val

with open(RESULTS, "a") as fh:
    best_str = pformat(best)
    fh.write(best_str)
    fh.write("\n")