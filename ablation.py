import logging
from copy import deepcopy
import os
import torch
import pandas
import argparse
from data import DVlog
from sam import SAM
from helpers import *
from mbt import MBT
from models import AblationModel
from torch.utils.data import DataLoader
from sklearn.metrics import recall_score, precision_score, accuracy_score, confusion_matrix

from helpers import is_jupyter
if is_jupyter():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


def train_sam(net, trainldr, optimizer, epoch, epochs, learning_rate, criteria):
    total_losses = AverageMeter()
    net.train()
    train_loader_len = len(trainldr)
    for batch_idx, data in enumerate(tqdm(trainldr)):
        feature_audio, feature_video, mask, labels = data

        # adjust_learning_rate(optimizer, epoch, epochs, learning_rate, batch_idx, train_loader_len)
        feature_audio = feature_audio.cuda()
        feature_video = feature_video.cuda()
        mask = mask.cuda()
        # labels = labels.float()
        labels = labels.cuda()
        optimizer.zero_grad()

        y = net(feature_audio, feature_video, mask)
        loss = criteria(y, labels)
        loss.backward()
        optimizer.first_step(zero_grad=True)

        y = net(feature_audio, feature_video, mask)
        criteria(y, labels).backward()
        optimizer.second_step(zero_grad=True)

        total_losses.update(loss.data.item(), feature_audio.size(0))
    return total_losses.avg()


def transform(y, yhat):
    i = np.argmax(yhat, axis=1)
    yhat = np.zeros(yhat.shape)
    yhat[np.arange(len(i)), i] = 1

    if not len(y.shape) == 1:
        if y.shape[1] == 1:
            y = y.reshape(-1)
        else:
            y = np.argmax(y, axis=-1)
    if not len(yhat.shape) == 1:
        if yhat.shape[1] == 1:
            yhat = yhat.reshape(-1)
        else:
            yhat = np.argmax(yhat, axis=-1)

    return y, yhat


def train(net, trainldr, optimizer, epoch, epochs, learning_rate, criteria):
    total_losses = AverageMeter()
    net.train()
    train_loader_len = len(trainldr)
    for batch_idx, data in enumerate(tqdm(trainldr)):
        # feature_audio, feature_video, labels = data
        feature_audio, feature_video, mask, labels = data

        # adjust_learning_rate(optimizer, epoch, epochs, learning_rate, batch_idx, train_loader_len)
        feature_audio = feature_audio.cuda()
        feature_video = feature_video.cuda()
        labels = labels.cuda()
        mask = mask.cuda()
        optimizer.zero_grad()

        y, bot_token = net(feature_audio, feature_video, mask)
        loss = criteria(y, labels)
        loss.backward()
        optimizer.step()

        total_losses.update(loss.data.item(), feature_audio.size(0))
    return total_losses.avg(), bot_token


def val(net, validldr, criteria):
    total_losses = AverageMeter()
    net.eval()
    all_y = None
    all_labels = None
    for batch_idx, data in enumerate(tqdm(validldr)):
        # feature_audio, feature_video, labels = data
        feature_audio, feature_video, mask, labels = data
        with torch.no_grad():
            feature_audio = feature_audio.cuda()
            feature_video = feature_video.cuda()
            mask = mask.cuda()
            # labels = labels.float()
            labels = labels.cuda()

            # y = net(feature_audio, feature_video)
            y, bot_token = net(feature_audio, feature_video, mask)
            loss = criteria(y, labels)
            total_losses.update(loss.data.item(), feature_audio.size(0))

            if all_y == None:
                all_y = y.clone()
                all_labels = labels.clone()
            else:
                all_y = torch.cat((all_y, y), 0)
                all_labels = torch.cat((all_labels, labels), 0)

    all_y = all_y.cpu().numpy()
    all_labels = all_labels.cpu().numpy()
    all_labels, all_y = transform(all_labels, all_y)

    f1 = f1_score(all_labels, all_y, average='weighted')
    r = recall_score(all_labels, all_y, average='weighted')
    p = precision_score(all_labels, all_y, average='weighted')
    acc = accuracy_score(all_labels, all_y)
    cm = confusion_matrix(all_labels, all_y)
    return (total_losses.avg(), f1, r, p, acc, cm, bot_token)


# def test(model, args, description):
#     keep = 'k' if args.keep else ''
#     testset = DVlog('{}test_{}{}.pickle'.format(args.datadir, keep, args.rate))
#     # testset = EmoDataset(args.val_manifest)
#     loss_fn = nn.CrossEntropyLoss()
#     # testldr = DataLoader(testset, batch_size=args.batch, collate_fn=new_collate_fn, shuffle=False, num_workers=1)
#     testldr = DataLoader(testset, batch_size=args.batch, collate_fn=collate_fn, shuffle=False, num_workers=1)

#     if not isinstance(model, nn.Module):
#         loaded = nn.DataParallel(MBT(25, 136, 256)).cuda()
#         # net = MBT(1*3, 10*3, 256, project_type=proj_type)
#         model = torch.load(model)
#         state_dict = model["state_dict"]
#         loaded.load_state_dict(state_dict)

#     eval_return = val(loaded, testldr, loss_fn)
#     print_eval_info(description, eval_return)



def main():
    exit()
    parser = argparse.ArgumentParser(description='Train task seperately')

    parser.add_argument('--net', '-n', default='mbt', help='Net name')
    parser.add_argument('--config', '-c', type=int, default=7, help='Config number')
    parser.add_argument('--batch', '-b', type=int, default=2, help='Batch size')
    parser.add_argument('--rate', '-R', default='4', help='Rate')
    parser.add_argument('--project', '-p', default='minimal', help='projection type')
    parser.add_argument('--epoch', '-e', type=int, default=100, help='Number of epoches')
    parser.add_argument('--lr', '-a', type=float, default=0.00001, help='Learning rate')
    parser.add_argument('--datadir', '-d', default='../../../Data/DVlog/', help='Data folder path')
    parser.add_argument('--train_manifest', '-t', default='audioset_less_train_manifest.csv', help='CSV file containing train manifest')
    parser.add_argument('--val_manifest', '-v', default='audioset_less_val_manifest.csv', help='CSV file containing val manifest')
    parser.add_argument('--sam', '-s', action='store_true', help='Apply SAM optimizer')
    parser.add_argument('--prenorm', '-P', action='store_true', help='Pre-norm')
    parser.add_argument('--keep', '-k', action='store_true', help='Keep all data in training set')

    args = parser.parse_args()
    # test("results/mbt7_1/best_val_acc_0.72549.pth", args, "Best Val Acc: ")
    # test("results/mbt7_1/best_val_f10.70989.pth", args, "Best Val F1: ")
    keep = 'k' if args.keep else ''
    output_dir = '{}{}_{}{}'.format(args.net, str(args.config), keep, args.rate)

    train_criteria = nn.CrossEntropyLoss()
    valid_criteria = nn.CrossEntropyLoss()

    trainset = DVlog('{}train_{}{}.pickle'.format(args.datadir, keep, args.rate))
    validset = DVlog('{}valid_{}{}.pickle'.format(args.datadir, keep, args.rate))
    trainldr = DataLoader(trainset, batch_size=args.batch, collate_fn=collate_fn, shuffle=True, num_workers=0)
    validldr = DataLoader(validset, batch_size=args.batch, collate_fn=collate_fn, shuffle=False, num_workers=0)
    proj_type = 'minimal'

    # trainset = EmoDataset(args.train_manifest)
    # validset = EmoDataset(args.val_manifest)
    # trainldr = DataLoader(trainset, batch_size=args.batch, collate_fn=new_collate_fn, shuffle=True, num_workers=1)
    # validldr = DataLoader(validset, batch_size=args.batch, collate_fn=new_collate_fn, shuffle=False, num_workers=1)
    # proj_type = 'conv2d'
    if args.net == 'mbt':
        # audio: 1 frame, 3 channels; video: 10 frames, 3 channels
        # net = MBT(1*3, 10*3, 256, project_type=proj_type)
        net = MBT(25, 136, 256)
    else:
        net = AblationModel(136, 25, 256, args.config, project_type=args.project, pre_norm=args.prenorm)
    net = nn.DataParallel(net).cuda()

    if args.sam:
        base_optimizer = torch.optim.SGD
        optimizer = SAM(net.parameters(), base_optimizer, lr=args.lr, momentum=0.9, weight_decay=1.0 / args.batch)
    else:
        optimizer = torch.optim.AdamW(net.parameters(), betas=(0.9, 0.999), lr=args.lr, weight_decay=1.0 / args.batch)

    best_f1 = 0.0
    best_acc = 0.0
    df = create_new_df()

    for epoch in range(args.epoch):
        if args.sam:
            train_loss = train_sam(net, trainldr, optimizer, epoch, args.epoch, args.lr, train_criteria)
        else:
            train_loss = train(net, trainldr, optimizer, epoch, args.epoch, args.lr, train_criteria)

        eval_return = val(net, validldr, valid_criteria)
        _, val_f1, _, _, val_acc, _ = eval_return
        description = "Epoch {:2d} | Rate {} | Trainloss {:.5f}:".format(epoch, args.rate, train_loss)
        print_eval_info(description, eval_return)

        os.makedirs(os.path.join('results', output_dir), exist_ok=True)

        if val_f1 >= best_f1:
            checkpoint = {'state_dict': net.state_dict()}
            torch.save(checkpoint, os.path.join('results', output_dir, f'best_val_f1{val_f1:.5f}.pth'))
            best_f1 = val_f1
            best_f1_model = deepcopy(net)

        if val_acc >= best_acc:
            checkpoint = {'state_dict': net.state_dict()}
            torch.save(checkpoint, os.path.join('results', output_dir, f'best_val_acc_{val_acc:.5f}.pth'))
            best_acc = val_acc
            best_acc_model = deepcopy(net)

        df = append_entry_df(df, eval_return)

    testset = DVlog('{}test_{}{}.pickle'.format(args.datadir, keep, args.rate))
    # testset = EmoDataset(args.val_manifest)
    test_criteria = nn.CrossEntropyLoss()
    # testldr = DataLoader(testset, batch_size=args.batch, collate_fn=new_collate_fn, shuffle=False, num_workers=1)
    testldr = DataLoader(testset, batch_size=args.batch, collate_fn=collate_fn, shuffle=False, num_workers=1)

    best_f1_model = nn.DataParallel(best_f1_model).cuda()
    eval_return = val(best_f1_model, testldr, test_criteria)
    description = 'Best F1 Testset'
    print_eval_info(description, eval_return)
    df = append_entry_df(df, eval_return)

    best_acc_model = nn.DataParallel(best_acc_model).cuda()
    eval_return = val(best_acc_model, testldr, test_criteria)
    description = 'Best Acc Testset'
    print_eval_info(description, eval_return)
    df = append_entry_df(df, eval_return)

    df = pandas.DataFrame(df)
    csv_name = os.path.join('results', output_dir, 'train.csv')
    df.to_csv(csv_name)


if __name__ == "__main__":
    main()
