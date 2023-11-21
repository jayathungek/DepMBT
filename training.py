import math
import sys
import pickle
from pprint import pformat
from pathlib import Path
import warnings
import argparse
from typing import List
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.nn import BCELoss
import torch.optim.lr_scheduler as lrsch

from cnn import LateFusionCNN, train_multicrop
from data import load_data
from constants import *
from helpers import ClassifierMetrics
from loss import multicrop_loss
from visualise import Visualiser
from datasets import emoreact, enterface


dataset_to_use = enterface

def get_best_checkpoint(checkpoints: List[str]):
    lowest = math.inf
    best = None
    for chkpt in checkpoints:
        checkpt_path = Path(chkpt)
        sections = checkpt_path.stem.split("_") 
        loss_val = float(sections[-1])
        if loss_val < lowest:
            lowest = loss_val
            best = checkpt_path.absolute()
    return best
    

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("script")
    parser.add_argument("-n", "--name")
    parser.add_argument("-t", "--visualise-train", default=False, action=argparse.BooleanOptionalAction)
    parsed = parser.parse_args(args)
    if parsed.name is None:
        print("Need model name: -n OR --name")
        exit(1)
    return parsed

def save_visualisation(model, dataloader, ds_namespace, best_chkpt, save_dir, is_train_dl=False):
    visualiser = Visualiser(ds_namespace)
    embeddings, labels = visualiser.get_embeddings_and_labels(dataloader, model)
    struct = {"embeddings": embeddings, "labels": labels}
    with open(f"{save_dir}/{best_chkpt['checkpoint_name']}{'_TRAIN_POINTS' if is_train_dl else ''}.pkl", "wb") as fh:
        pickle.dump(struct, fh)

    with open(RESULTS, "a") as fh:
        best_str = pformat(best_chkpt)
        fh.write(best_str)
        fh.write("\n")


if __name__ == "__main__":
    mbt_teacher = LateFusionCNN(enterface)
    mbt_teacher = nn.DataParallel(mbt_teacher).cuda()

    mbt_student = LateFusionCNN(enterface)
    mbt_student = nn.DataParallel(mbt_student).cuda()

    train_dl, val_dl, test_dl, split_seed = load_data(dataset_to_use, 
                                        batch_sz=BATCH_SZ,
                                        train_val_test_split=SPLIT, 
                                        force_audio_aspect=FORCE_AUDIO_ASPECT)

    # only the student's weights are updated by the optimiser
    optimizer = torch.optim.AdamW(mbt_student.parameters(), betas=BETAS, lr=LR, weight_decay=WEIGHT_DECAY)

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
        "weight_decay": WEIGHT_DECAY,
        "betas": BETAS,
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
        "split_seed": split_seed,
        "force_audio_aspect": FORCE_AUDIO_ASPECT,
        "best_epoch": 0,
        "val": {
            "loss": None,
        },
        "train": {
            "loss": None,
        },
        "checkpoint_name": None
    }

    parsed_args = parse_args(sys.argv)
    experiment_name = parsed_args.name
    save_path = Path(f"saved_models/{experiment_name}")
    if save_path.exists() and save_path.is_dir():
        print(f"Experiment exists, searching for checkpoints...")
        checkpoints = list(save_path.glob("*pth"))
        if len(checkpoints) > 0:
            best_checkpoint = get_best_checkpoint(checkpoints)
            save_items = torch.load(best_checkpoint)
            teacher_state_dict = save_items["teacher_state_dict"]
            student_state_dict = save_items["student_state_dict"]
            mbt_teacher = mbt_teacher.load_state_dict(teacher_state_dict)
            mbt_student = mbt_student.load_state_dict(student_state_dict)
            updated_centre = save_items["centre"]
        else:
            print("No checkpoints found, training from scratch.")
            updated_centre = CENTRE_CONSTANT
    else:
        save_path.mkdir(exist_ok=False)
        updated_centre = CENTRE_CONSTANT

    fpath_params = save_path / "hparams.txt"
    with fpath_params.open("w") as fh:
        best_str = pformat(best)
        fh.write(best_str)
        fh.write("\n")
    try:
        for epoch in range(EPOCHS):
            train_loss, updated_centre = train_multicrop(mbt_teacher, mbt_student, train_dl, optimizer, updated_centre, loss_fn=multicrop_loss)
            scheduler.step()
            print(f"Epoch {epoch + 1}: train_loss {train_loss:.5f}\n")

            if best["train"]["loss"] is None or (best["train"]["loss"] is not None and train_loss <= best["train"]["loss"]): 
                fname = f"mbt_student_train_loss_{train_loss:.5f}"
                fpath_chkpt = save_path / f"{fname}.pth"
                best["T_0"] = T_0
                best["best_epoch"] = epoch + 1
                best["train"]["loss"] = train_loss
                best["checkpoint_name"] = fname
                save_items = {
                    "student_state_dict": mbt_student.state_dict(), 
                    "teacher_state_dict": mbt_teacher.state_dict(), 
                    "centre": updated_centre
                }
                torch.save(save_items, fpath_chkpt)
    except (Exception, KeyboardInterrupt) as e:
        print(f"Fatal error: {e}")
    finally:
        print(pformat(best))
        if parsed_args.visualise_train:
            print(f"Saving visualisation for best checkpoint {best['checkpoint_name']} (model tested on train set)")
            save_visualisation(mbt_student, train_dl, dataset_to_use, best, save_path, parsed_args.visualise_train)
        else:
            print(f"Saving visualisation for best checkpoint {best['checkpoint_name']} (model tested on val set)")
            save_visualisation(mbt_student, val_dl, dataset_to_use, best, save_path, parsed_args.visualise_train)
