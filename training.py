import torch
from torch.utils.data import DataLoader
from torch.nn import BCELoss
import torch.nn as nn


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
from data import EmoDataset, new_collate_fn
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
vmbt = ViTMBT(1024, num_class=LABELS)
vmbt = nn.DataParallel(vmbt).cuda()

se = None
nlines = None
train_ds = EmoDataset(f"{DATA_DIR}/Labels/train_labels_full_pruned.csv", nlines=nlines, sole_emotion=se)
val_ds = EmoDataset(f"{DATA_DIR}/Labels/val_labels_full_pruned.csv", nlines=nlines, sole_emotion=se)
test_ds = EmoDataset(f"{DATA_DIR}/Labels/test_labels_full_pruned.csv", nlines=nlines, sole_emotion=se)

train_metrics = ClassifierMetrics(task='multilabel', n_labels=LABELS, device=DEVICE)
val_metrics = ClassifierMetrics(task='multilabel', n_labels=LABELS, device=DEVICE)

train_dl = DataLoader(train_ds, collate_fn=new_collate_fn, batch_size=BATCH_SZ, shuffle=True)
val_dl = DataLoader(val_ds, collate_fn=new_collate_fn, batch_size=BATCH_SZ, shuffle=False)
test_dl = DataLoader(test_ds, collate_fn=new_collate_fn, batch_size=BATCH_SZ, shuffle=False)

optimizer = torch.optim.AdamW(vmbt.parameters(), betas=(0.9, 0.999), lr=lr, weight_decay=1.0 / BATCH_SZ)
# optimizer = torch.optim.SGD(vmbt.parameters(), lr=lr, momentum=momentum, weight_decay=1/BATCH_SZ)

loss_func = BCELoss()
# loss_func = nn.CrossEntropyLoss()

best = {
    "optim": {optimizer.__class__.__name__},
    "optim_lr": {optimizer.defaults.get('lr')},
    "optim_momentum": {optimizer.defaults.get('momentum')},
    "optim_weight_decay": {optimizer.defaults.get('weight_decay')},
    "loss": {loss_func.__class__.__name__},
    "batch_sz": {BATCH_SZ},
    "epochs": {EPOCHS},
    "val_loss": None,
    "val_f1": None,
    "val_recall": None,
    "val_precision": None,
    "val_acc": None
}

print(f"optim: {optimizer.__class__.__name__}")
print(f"optim lr: {optimizer.defaults.get('lr')}")
print(f"optim momentum: {optimizer.defaults.get('momentum')}")
print(f"loss: {loss_func.__class__.__name__}")
print(f"batch sz: {BATCH_SZ}")
print(f"epochs: {EPOCHS}")


for epoch in range(EPOCHS):
    train_metrics = train(vmbt, train_dl, optimizer, loss_fn=loss_func, cls_metrics=train_metrics)
    # train_loss, train_f1, train_r, train_p, train_acc, train_bottleneck_tokens = train_metrics
    train_loss, train_f1, train_r, train_p, train_acc = train_metrics
    val_metrics = val(vmbt, val_dl, loss_fn=loss_func, cls_metrics=val_metrics)
    # val_loss, val_f1, val_r, val_p, val_acc, val_bottleneck_tokens = val_metrics
    val_loss, val_f1, val_r, val_p, val_acc = val_metrics
    # train_token = train_bottleneck_tokens[0][:, :256]
    # val_token = val_bottleneck_tokens[0][:, :256]

    print(
        (f"Epoch {epoch + 1}: train_loss {train_loss:.5f}, val_loss {val_loss:.5f}\n"
            f"                   train_precision {train_p}, val_precision {val_p}\n"
            f"                   train_recall {train_r}, val_recall {val_r}\n"
            f"                   train_f1 {train_f1}, val_f1 {val_f1}"
            )
        )
    if best["val_f1"] is None or (best["val_f1"] is not None and val_f1 > best["val_f1"]): 
        best["val_loss"] = val_loss
        best["val_f1"] = val_f1
        best["val_recall"] = val_r
        best["val_precision"] = val_p
        best["val_acc"] = val_acc

    # display_tensor_as_rgb(train_token, "Train bottleneck token") 
    
with open(RESULTS, "a") as fh:
    fh.write(str(best))
    fh.write("\n")