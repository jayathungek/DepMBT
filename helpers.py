from turtle import forward
import torch
import numpy as np
import torch.nn as nn
import torch 
from math import cos, pi
from sklearn.metrics import f1_score
import torchmetrics.classification as tmcls 
import matplotlib as mblib
import matplotlib.pyplot as plt
from itertools import repeat
import collections.abc

EPS = 1e-8

# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def make_divisible(v, divisor=8, min_value=None, round_limit=.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


def extend_tuple(x, n):
    # pdas a tuple to specified n by padding with last value
    if not isinstance(x, (tuple, list)):
        x = (x,)
    else:
        x = tuple(x)
    pad_n = n - len(x)
    if pad_n <= 0:
        return x[:n]
    return x + (x[-1],) * pad_n



class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        n = float(n)
        self.sum += val * n
        self.count += n
    
    def avg(self):
        return (self.sum / self.count)

def adjust_learning_rate(optimizer, epoch, epochs, init_lr, iteration, num_iter):

    current_iter = iteration + epoch * num_iter
    max_iter = epochs * num_iter
    lr = init_lr * (1 + cos(pi * current_iter / max_iter)) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def load_state_dict(model,path):
    checkpoints = torch.load(path,map_location=torch.device('cpu'))
    state_dict = checkpoints['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module.' in k:
            k = k[7:]  # remove `module.`
        new_state_dict[k] = v
    # load params
    model.load_state_dict(new_state_dict,strict=False)
    return model

class WeightedAsymmetricLoss(nn.Module):
    def __init__(self, eps=1e-8, disable_torch_grad=True, weight=None):
        super(WeightedAsymmetricLoss, self).__init__()
        self.disable_torch_grad = disable_torch_grad
        self.eps = eps
        self.weight = weight

    def forward(self, x, y, mask):
        xs_pos = x
        xs_neg = 1 - x
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))

        if self.disable_torch_grad:
            torch.set_grad_enabled(False)
        neg_weight = 1 - xs_neg
        if self.disable_torch_grad:
            torch.set_grad_enabled(True)
        loss = los_pos + neg_weight * los_neg

        if self.weight is not None:
            loss = loss * self.weight.view(1,-1)

        loss = loss.mean(dim=-1)
        loss = loss * mask
        return -loss.sum() / (mask.sum() + EPS)

class RegressionLoss(nn.Module):
    def __init__(self):
        super(RegressionLoss, self).__init__() 
        self.loss = MaskNegativeCCCLoss()

    def forward(self, x, y, mask):
        loss1 = self.loss(x[:, 0], y[:, 0], mask) + self.loss(x[:, 1], y[:, 1], mask)
        return loss1

class MaskedCELoss(nn.Module):
    def __init__(self, weight=None, ignore_index=-1):
        super(MaskedCELoss, self).__init__() 
        self.ce = nn.CrossEntropyLoss(reduction='none', weight=weight, ignore_index=ignore_index)
    
    def forward(self, x, y, mask):
        loss = self.ce(x, y)
        loss = loss.mean(dim=-1)
        loss = loss * mask
        return loss.sum() / (mask.sum() + EPS)

class DistillationLoss(nn.Module):
    def __init__(self, alpha, func):
        super(DistillationLoss, self).__init__() 
        self.alpha = alpha
        self.func = func
    
    def forward(self, yhat, ysoft, y, mask):
        return (1 - self.alpha) * self.func(yhat, ysoft, mask) + self.alpha * self.func(yhat, y, mask)


class DistillationLossFromLogit(nn.Module):
    def __init__(self, alpha, func, temp):
        super(DistillationLossFromLogit, self).__init__() 
        self.alpha = alpha
        self.func = func
        self.temp = temp
        self.distill_loss = nn.KLDivLoss(log_target=True, reduction='batchmean')
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, yhat, ysoft, y, mask):
        return self.distill_loss(
            torch.log(self.softmax(ysoft / self.temp)),
            torch.log(self.softmax(yhat / self.temp))
        ) * self.temp**2



class MaskNegativeCCCLoss(nn.Module):
    def __init__(self):
        super(MaskNegativeCCCLoss, self).__init__()
    def forward(self, x, y, m):
        y = y.view(-1)
        x = x.view(-1)
        x = x * m
        y = y * m
        N = torch.sum(m)
        x_m = torch.sum(x) / N
        y_m = torch.sum(y) / N
        vx = (x - x_m) * m
        vy = (y - y_m) * m
        ccc = 2*torch.dot(vx, vy) / (torch.dot(vx, vx) + torch.dot(vy, vy) + N * torch.pow(x_m - y_m, 2) + EPS)
        return 1-ccc

def CCC_score(x, y):
    vx = x - np.mean(x)
    vy = y - np.mean(y)
    rho = np.sum(vx * vy) / (np.sqrt(np.sum(vx**2)) * np.sqrt(np.sum(vy**2)))
    x_m = np.mean(x)
    y_m = np.mean(y)
    x_s = np.std(x)
    y_s = np.std(y)
    ccc = 2*rho*x_s*y_s/(x_s**2 + y_s**2 + (x_m - y_m)**2)
    return ccc

def EX_metric(y, yhat):
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

    return f1_score(y, yhat, average='macro'), f1_score(y, yhat, average=None)


def VA_metric(y, yhat):
    cccs = (float(CCC_score(y[:,0], yhat[:,0])), float(CCC_score(y[:,1], yhat[:,1])))
    avg_ccc = float(CCC_score(y[:,0], yhat[:,0]) + CCC_score(y[:,1], yhat[:,1])) / 2
    return avg_ccc, cccs


def AU_metric(y, yhat, thresh=0.5):
    yhat = (yhat >= thresh)
    N, label_size = y.shape
    f1s = []
    for i in range(label_size):
        f1 = f1_score(y[:, i], yhat[:, i])
        f1s.append(f1)
    return np.mean(f1s), f1s

def normalize_digraph(A):
    b, n, _ = A.shape
    node_degrees = A.detach().sum(dim = -1)
    degs_inv_sqrt = node_degrees ** -0.5
    norm_degs_matrix = torch.eye(n)
    dev = A.get_device()
    if dev >= 0:
        norm_degs_matrix = norm_degs_matrix.to(dev)
    norm_degs_matrix = norm_degs_matrix.view(1, n, n) * degs_inv_sqrt.view(b, n, 1)
    norm_A = torch.bmm(torch.bmm(norm_degs_matrix,A),norm_degs_matrix)
    return norm_A

def append_entry_df(df, entry):
    val_loss, val_f1, val_recall, val_precision, val_acc, val_matrix = entry
    df['val_loss'].append(val_loss)
    df['val_f1'].append(val_f1)
    df['val_acc'].append(val_acc)
    df['val_recall'].append(val_recall)
    df['val_precision'].append(val_precision)
    df['val_tn'].append(val_matrix[0][0])
    df['val_fp'].append(val_matrix[0][1])
    df['val_fn'].append(val_matrix[1][0])
    df['val_tp'].append(val_matrix[1][1])
    return df

def create_new_df():
    df =  {}
    df['val_loss'] = []
    df['val_f1'] = []
    df['val_recall'] = []
    df['val_precision'] = []
    df['val_acc'] = []
    df['val_tn'] = []
    df['val_fp'] = []
    df['val_fn'] = []
    df['val_tp'] = []
    return df

def print_eval_info(description, eval_return):
    val_loss, val_f1, val_recall, val_precision, val_acc, val_matrix = eval_return
    print('{}: {:.5f},{:.5f},{:.5f},{:.5f},{:.5f}'.format(description, val_loss, val_acc, val_recall, val_precision, val_f1))
    print('Confusion matrix {} {} {} {}'
            .format(val_matrix[0][0],
                    val_matrix[0][1],
                    val_matrix[1][0],
                    val_matrix[1][1]))


def logits_to_labels(logits, threshold=0.5):
    return torch.where(logits > threshold, 1.0, 0.0)


class ClassifierMetrics(object):
    ap: float
    precision: float
    recall: float
    f1: float
    acc: float
    count: int # number of batches seen 

    def __init__(self, task, n_labels, device):
        self.task = task
        if self.task == "multiclass":
            self.ap_metric = tmcls.MulticlassAveragePrecision(num_classes=n_labels, average=None, thresholds=None).to(device)
            self.precision_metric = tmcls.MulticlassPrecision(num_classes=n_labels).to(device)
            self.recall_metric = tmcls.MulticlassRecall(num_classes=n_labels).to(device)
            self.f1_metric = tmcls.MulticlassF1Score(num_classes=n_labels).to(device)
            self.acc_metric = tmcls.MulticlassAccuracy(num_classes=n_labels).to(device)

        elif self.task == "multilabel":
            self.ap_metric = tmcls.MultilabelAveragePrecision(num_labels=n_labels, average=None, thresholds=None).to(device)
            self.precision_metric = tmcls.MultilabelPrecision(num_labels=n_labels).to(device)
            self.recall_metric = tmcls.MultilabelRecall(num_labels=n_labels).to(device)
            self.f1_metric = tmcls.MultilabelF1Score(num_labels=n_labels).to(device)
            self.acc_metric = tmcls.MultilabelAccuracy(num_labels=n_labels).to(device)
        self.reset()
    

    def reset(self):
        self.ap = 0
        self.precision = 0
        self.recall = 0
        self.f1 = 0
        self.acc = 0
        self.count = 0

    def update(self, y_pred, y):
        y = y.long()
        self.ap += self.ap_metric(y_pred, y)
        self.precision += self.precision_metric(y_pred, y)
        self.recall += self.recall_metric(y_pred, y)
        self.f1 += self.f1_metric(y_pred, y)
        self.acc += self.acc_metric(y_pred, y)
        self.count += 1 

    def avg(self):
        self.ap = self.ap / self.count
        self.precision = self.precision / self.count
        self.recall = self.recall / self.count
        self.f1 = self.f1 / self.count
        self.acc = self.acc / self.count
        


def display_tensor_as_rgb(tensor, title=None):
    tensor = tensor.cpu().detach().numpy()
    t_y = tensor.shape[0]
    t_x = tensor.shape[1]

    cur_aspect_ratio = t_y/t_x
    if cur_aspect_ratio <= 1:
        aspect = 1/cur_aspect_ratio
    else:
        aspect = cur_aspect_ratio
    plt.figure(frameon=False)
    plt.title(title if title else '')
    plt.tick_params(bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

    plt.imshow(tensor, aspect=aspect)
    plt.colorbar()
    plt.show()



def is_jupyter():
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is None:
            # we have IPython installed but not running from IPython
            return False
        else:
            return True
    except:
        # We do not even have IPython installed
        return False

if __name__ == "__main__":
    print(is_jupyter())