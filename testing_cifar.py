from pympler.tracker import SummaryTracker
tracker = SummaryTracker()
from time import sleep
import torch
from torch.utils.data import DataLoader, random_split
from torch.nn import CrossEntropyLoss
import torch.nn as nn


from importlib import reload

import helpers
import vitunimodal
reload(helpers)
reload(vitunimodal)
from helpers import ClassifierMetrics, display_tensor_as_rgb
from vitunimodal import ViTMBT, train, val
import warnings
import multiprocessing as mp

from torchvision.datasets import CIFAR10
from torchvision.transforms.functional import pil_to_tensor, resize
warnings.filterwarnings('ignore')

from constants import *

HEIGHT = WIDTH = 32
NUM_LABELS = 10

def collate_fn(batch):
    rgb_batch_tensor = torch.FloatTensor(len(batch), 3, HEIGHT, WIDTH)
    rgb_tensor_list = []
    labels_list = []
    last_file, last_label =  None, None
    for pil_image, label in batch:
        rgb = resize(pil_to_tensor(pil_image),(224, 224)).unsqueeze(0)
        rgb_tensor_list.append(rgb)
        labels_list.append(label)

    torch.cat(rgb_tensor_list, out=rgb_batch_tensor)
    label_batch_tensor = torch.LongTensor(labels_list)

    return rgb_batch_tensor, label_batch_tensor

def load_data(data_path, batch_sz=64, train_val_test_split=[0.7, 0.1, 0.2]):
    # This is a convenience funtion that returns dataset splits of train, val and test according to the fractions specified in the arguments
    assert sum(train_val_test_split) == 1, "Train, val and test fractions should sum to 1!"  # Always a good idea to use static asserts when processing arguments that are passed in by a user!
    dataset = CIFAR10(data_path)  # Instantiating our previously defined dataset
    
    # This code generates the actual number of items that goes into each split using the user-supplied fractions
    tr_va_te = []
    for frac in train_val_test_split:
        actual_count = frac * len(dataset)
        actual_count = round(actual_count)
        tr_va_te.append(actual_count)
    
    # split dataset into train, val and test
    train_split, val_split, test_split = random_split(dataset, tr_va_te)
    
    # Use Pytorch DataLoader to load each split into memory. It's important to pass in our custom collate function, so it knows how to interpret the 
    # data and load it. num_workers tells the DataLoader how many CPU threads to use so that data can be loaded in parallel, which is faster
    # n_cpus = mp.cpu_count() # returns number of CPU cores on this machine
    n_cpus = 8
    train_dl = DataLoader(train_split, 
                          batch_size=batch_sz, 
                          shuffle=True, 
                          collate_fn=collate_fn,
                          num_workers=n_cpus)            
    val_dl = DataLoader(val_split, 
                        batch_size=batch_sz, 
                        shuffle=True, 
                        collate_fn=collate_fn,
                        num_workers=n_cpus)
    test_dl = DataLoader(test_split,
                         batch_size=batch_sz,
                         shuffle=False,
                         collate_fn=collate_fn,
                         num_workers=n_cpus)
    return train_dl, val_dl, test_dl

train_dl, val_dl, test_dl = load_data("/root/intelpa-2/datasets/cifar10", train_val_test_split=[0.8, 0.1, 0.1])
for data, label in train_dl:
    print(data.shape)
    break


DEVICE = "cuda"
RESULTS = "results/best.txt"
PRETRAINED_CHKPT = "./pretrained_models/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz"
EPOCHS = 15
lr = 0.00001
betas = (0.9, 0.999)
momentum = 0.9
BATCH_SZ = 16
LABELS = 10
vmbt = ViTMBT(1024, num_class=LABELS)
vmbt = nn.DataParallel(vmbt).cuda()


train_cls = ClassifierMetrics(task='multiclass', n_labels=LABELS, device=DEVICE)
val_cls = ClassifierMetrics(task='multiclass', n_labels=LABELS, device=DEVICE)


optimizer = torch.optim.AdamW(vmbt.parameters(), betas=(0.9, 0.999), lr=lr, weight_decay=1.0 / BATCH_SZ)

loss_func = CrossEntropyLoss()

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

    if best["val_f1"] is None or (best["val_f1"] is not None and val_f1_val > best["val_f1"]): 
        best["val_loss"] = val_loss
        best["val_f1"] = val_f1_val
        best["val_recall"] = val_recall_val
        best["val_precision"] = val_precision_val
        best["val_acc"] = val_acc_val

with open(RESULTS, "a") as fh:
    fh.write(str(best))
    fh.write("\n")

tracker.print_diff()    