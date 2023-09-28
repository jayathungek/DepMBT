import time
import copy
import logging
import dataclasses

from typing import Optional, Dict, Callable

from PIL import Image
import PIL.ImageOps
import torch
import torchvision.transforms as T
import torch.nn as nn
from torch.nn import BCELoss
from torchvision import transforms
from torch.utils.data import DataLoader
from timm.models._manipulate import adapt_input_conv
from timm.models._helpers import load_state_dict
from timm.data.transforms_factory import create_transform
from timm.models._pretrained import PretrainedCfg
from timm.models._registry import get_pretrained_cfg
from timm.data import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from sklearn.metrics import recall_score, precision_score, accuracy_score, confusion_matrix

from helpers import *
from patch_embed import PatchEmbed
from data import EmoDataset, new_collate_fn
from constants import *
from vision_transformer import VisionTransformer, Block
from ablation import transform
from patch_embed import PatchEmbed

from helpers import is_jupyter
if is_jupyter():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm




_logger = logging.getLogger(__name__)


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


def load_pretrained(
        model: nn.Module,
        pretrained_cfg: Optional[Dict] = None,
):
    """ Load pretrained checkpoint

    Args:
        model (nn.Module) : PyTorch model module
        pretrained_cfg (Optional[Dict]): configuration for pretrained weights / target dataset
        num_classes (int): num_classes for target model
        in_chans (int): in_chans for target model
        filter_fn (Optional[Callable]): state_dict filter fn for load (takes state_dict, model as args)
        strict (bool): strict load of checkpoint

    """
    pretrained_cfg = pretrained_cfg or getattr(model, 'pretrained_cfg', None)
    if not pretrained_cfg:
        _logger.warning("Invalid pretrained config, cannot load weights.")
        return

    pretrained_loc = pretrained_cfg["url"]
    _logger.info(f'Loading pretrained weights from file ({pretrained_loc})')
    model.load_pretrained(pretrained_loc)



class PretrainedAST(nn.Module):
    def __init__(self, pretrained_checkpoint_path, cutoff_layer, channels, model_name, no_class, drop):
        super().__init__()
        print(f"Loading {model_name}...")
        t_start = time.time()
        cfg = _cfg(pretrained_checkpoint_path)

        self.model = VisionTransformer(in_chans=channels, no_embed_class=no_class, img_size=(NUM_MELS, MAX_SPEC_SEQ_LEN), attn_drop_rate=drop)
        load_pretrained(self.model, pretrained_cfg=cfg)
        delta_t = time.time() - t_start
        print(f"Loaded successfully in {delta_t:.1f}s")

        trimmed_layers = self.model.blocks[:cutoff_layer]
        del self.model.norm
        del self.model.fc_norm
        del self.model.head
        self.model.blocks = trimmed_layers

    def forward(self, x):
        return self.model.forward_features(x)

class PretrainedViT(nn.Module):
    def __init__(self, pretrained_checkpoint_path, cutoff_layer, channels, model_name, no_class, drop, embed_layer=PatchEmbed, img_size=(224, 224)):
        super().__init__()
        print(f"Loading {model_name}...")
        t_start = time.time()
        cfg = _cfg(pretrained_checkpoint_path)
        self.model = VisionTransformer(img_size=img_size, in_chans=channels, no_embed_class=no_class, embed_layer=embed_layer, attn_drop_rate=drop)
        load_pretrained(self.model, pretrained_cfg=cfg)
        delta_t = time.time() - t_start
        print(f"Loaded successfully in {delta_t:.1f}s")

        trimmed_layers = self.model.blocks[:cutoff_layer]
        del self.model.norm
        del self.model.fc_norm
        del self.model.head
        self.model.blocks = trimmed_layers


    def forward(self, x):
        return self.model.forward_features(x)
        

def reset_weights(m: nn.Module, cutoff):
    for name, module in m.named_modules():
        if name:
            
            try:
                layer = int(name)
            except ValueError:
                layer = int(name.split('.')[0])

            if layer >= cutoff:
                if hasattr(module, 'reset_parameters'): 
                    print(f'Reset trainable parameters of layer = {name}')
                    module.reset_parameters()

# try and implement DropDim regularisation (https://ieeexplore.ieee.org/document/9670702)

class ViTMBT(nn.Module):
    def __init__(self, embed_dim, num_bottle_token=4, bottle_layer=12
                , project_type='conv2d', num_head=4, attn_drop=0.1, linear_drop=0.1, pt_attn_drop=0.1, num_layers=14, num_class=8, no_class=True, freeze_first=10, apply_augmentation=False):
        super().__init__()
        self.num_class = num_class
        self.classification_threshold = 0.75
        self.num_modalities = 2
        self.num_layers = num_layers
        self.bottle_layer = bottle_layer
        self.num_bottle_token = num_bottle_token
        self.num_multimodal_layers = num_layers - bottle_layer
        # make vision transformer layers be accessible via subscript
        self.unimodal_audio = PretrainedAST(PRETRAINED_CHKPT, self.bottle_layer, 3, "audio layers", drop=pt_attn_drop, no_class=no_class)
        self.unimodal_video = PretrainedViT(PRETRAINED_CHKPT, self.bottle_layer, 3 * FRAMES, "video layers", drop=pt_attn_drop, no_class=no_class)
        self.multimodal_audio = clones(Block(embed_dim, num_head, attn_drop=attn_drop, qk_norm=True), self.num_multimodal_layers)
        self.multimodal_video = clones(Block(embed_dim, num_head, attn_drop=attn_drop, qk_norm=True), self.num_multimodal_layers)
        self.bottleneck_token = nn.Parameter(torch.zeros(1, num_bottle_token, embed_dim))
        self.acls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.vcls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.hidden_sz = 512
        self.head = nn.Sequential(
            nn.Linear(self.num_modalities * embed_dim, self.hidden_sz), # there are 2 modalities, each with embed_dim features
            nn.Dropout(linear_drop),
            nn.Linear(self.hidden_sz, self.num_class)
        ) 
        self.sigmoid = nn.Sigmoid()
        self.video_augmentations = nn.Sequential(
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.RandomInvert(p=0.2)
        ) if apply_augmentation else nn.Identity()

        assert freeze_first < self.bottle_layer, f"freeze_first must be at most the number of layers until the bottleneck layer: {self.bottle_layer}"
        for layer_num in range(freeze_first):
            self.unimodal_audio.model.blocks[layer_num].requires_grad_(False)
            self.unimodal_video.model.blocks[layer_num].requires_grad_(False)



        reset_weights(self.unimodal_audio.model.blocks, freeze_first)
        reset_weights(self.unimodal_video.model.blocks, freeze_first)

    def forward(self, a, v):
        B = a.shape[0] # Batch size
        a = self.unimodal_audio.model.embed_project(a)
        v = self.unimodal_video.model.embed_project(v)

        a = self.unimodal_audio.model.forward_features(a)
        v = self.unimodal_video.model.forward_features(v)


        bottleneck_token = self.bottleneck_token.expand(B, -1, -1)

        for i in range(self.num_multimodal_layers):
            a = torch.cat((bottleneck_token, a), dim=1)
            v = torch.cat((bottleneck_token, v), dim=1)

            a  = self.multimodal_audio[i](a)
            v  = self.multimodal_video[i](v)

            bottleneck_token = (a[:, :self.num_bottle_token] + v[:, :self.num_bottle_token]) / 2
            a = a[:, self.num_bottle_token:]
            v = v[:, self.num_bottle_token:]


        out = torch.cat((a[:, :1, :], v[:, :1, :]), dim=1) # concatenating the classification tokens
        out = out.flatten(start_dim=1, end_dim=2)
        out = self.head(out)
        out = out.reshape(B, 1, self.num_class)
        out = self.sigmoid(out)
        return out, bottleneck_token

def train(net, trainldr, optimizer, loss_fn, cls_metrics):
    total_losses = AverageMeter()
    net.train()
    cls_metrics.reset()
    for data in tqdm(trainldr):
        audio, video, labels = data
        batch_sz = video.shape[0]
        video = video.reshape(batch_sz * 10, 3, 224, 224)
        video = net.module.video_augmentations(video)
        video = video.reshape(batch_sz, 30, 224, 224)
        audio = audio.to(DEVICE)
        video = video.to(DEVICE)
        labels = labels.float()
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        y, bot_token = net(audio, video)
        loss = loss_fn(y, labels)
        y = (y > net.module.classification_threshold).float()
        cls_metrics.update(y.squeeze(1), labels.squeeze(1))
        loss.backward()
        optimizer.step()
        total_losses.update(loss.data.item(), audio.size(0))

    cls_metrics.avg()
    return total_losses.avg(), cls_metrics

def val(net, valldr, loss_fn, cls_metrics):
    total_losses = AverageMeter()
    cls_metrics.reset()
    net.eval()
    with torch.no_grad():
        for data in tqdm(valldr):
            audio, video, labels = data
            batch_sz = video.shape[0]
            audio = audio.to(DEVICE)
            video = video.to(DEVICE)
            labels = labels.float()
            labels = labels.to(DEVICE)

            y, bot_token = net(audio, video)
            loss = loss_fn(y, labels)

            y = (y > net.module.classification_threshold).float()
            cls_metrics.update(y.squeeze(1), labels.squeeze(1))
            total_losses.update(loss.data.item(), audio.size(0))
    cls_metrics.avg()
    return total_losses.avg(), cls_metrics

            


def toy_test(model):
    config = {
        'input_size': (3, 224, 224), 
        'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 
        'std': (0.5, 0.5, 0.5),
        'crop_pct': 0.9, 'crop_mode': 'center'
    }
    transform = create_transform(**config)
    img = Image.open("dog.jpg").convert('RGB')
    audio_batch = transform(img).unsqueeze(0)  
    video_batch = transform(img).unsqueeze(0)  
    output = model(audio_batch, video_batch)

class ViTAudio(nn.Module):
    def __init__(self, embed_dim, bottle_layer=12, num_head=4, num_layers=14, num_class=8, no_class=False, freeze_first=10, attn_drop=0.1, linear_drop=0.1, pt_attn_drop=0.1):
        super().__init__()
        self.num_class = num_class
        self.cutoff_layer = bottle_layer
        self.classification_threshold = 0.75
        self.num_modalities = 1
        self.num_multimodal_layers = num_layers - self.cutoff_layer
        # self.unimodal_audio = PretrainedViT(PRETRAINED_CHKPT, self.cutoff_layer, 3, "audio layers", no_class=no_class)
        self.unimodal_audio = PretrainedAST(PRETRAINED_CHKPT, self.cutoff_layer, 3, "audio layers", no_class=no_class, drop=pt_attn_drop)
        self.multimodal_audio = clones(Block(embed_dim, num_head, attn_drop=attn_drop, qk_norm=True), self.num_multimodal_layers)
        self.acls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.hidden_sz = 512
        self.head = nn.Sequential(
            nn.Linear(self.num_modalities * embed_dim, self.hidden_sz), # there are 2 modalities, each with embed_dim features
            nn.Dropout(linear_drop),
            nn.Linear(self.hidden_sz, self.num_class)
        ) 
        self.sigmoid = nn.Sigmoid()

        assert freeze_first <= self.cutoff_layer, f"freeze_first must be at most the number of layers until the cutoff layer: {self.cutoff_layer}"
        for layer_num in range(freeze_first):
            self.unimodal_audio.model.blocks[layer_num].requires_grad_(False)


        reset_weights(self.unimodal_audio.model.blocks, freeze_first)

    def forward(self, a):
        B = a.shape[0] # Batch size
        a = self.unimodal_audio.model.embed_project(a)
        a = self.unimodal_audio.model.forward_features(a)

        for i in range(self.num_multimodal_layers):
            a  = self.multimodal_audio[i](a)

        out = a[:, :1, :] 
        out = out.flatten(start_dim=1, end_dim=2)
        out = self.head(out)
        out = out.reshape(B, 1, self.num_class)
        out = self.sigmoid(out)
        return out

def train_audio(net, trainldr, optimizer, loss_fn, cls_metrics):
    total_losses = AverageMeter()
    net.train()
    cls_metrics.reset()
    for data in tqdm(trainldr):
        audio, _, labels = data
        audio = audio.to(DEVICE)
        labels = labels.float()
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        y = net(audio)
        loss = loss_fn(y, labels)
        y = (y > net.module.classification_threshold).float()
        cls_metrics.update(y.squeeze(1), labels.squeeze(1))
        loss.backward()
        optimizer.step()
        total_losses.update(loss.data.item(), audio.size(0))

    cls_metrics.avg()
    return total_losses.avg(), cls_metrics

def val_audio(net, valldr, loss_fn, cls_metrics):
    total_losses = AverageMeter()
    cls_metrics.reset()
    net.eval()
    with torch.no_grad():
        for data in tqdm(valldr):
            audio, _, labels = data
            audio = audio.to(DEVICE)
            labels = labels.float()
            labels = labels.to(DEVICE)

            y = net(audio)
            loss = loss_fn(y, labels)

            y = (y > net.module.classification_threshold).float()
            cls_metrics.update(y.squeeze(1), labels.squeeze(1))
            total_losses.update(loss.data.item(), audio.size(0))
    cls_metrics.avg()
    return total_losses.avg(), cls_metrics


class ViTVideo(nn.Module):
    def __init__(self, embed_dim, bottle_layer=12, num_head=4, attn_drop=0.1, linear_drop=0.1, pt_attn_drop=0.1, num_layers=14, num_class=8, no_class=False, freeze_first=10, apply_augmentation=False):
        super().__init__()
        self.num_class = num_class
        self.cutoff_layer = bottle_layer
        self.classification_threshold = 0.75
        self.num_modalities = 1
        self.num_multimodal_layers = num_layers - self.cutoff_layer
        self.unimodal_video = PretrainedViT(PRETRAINED_CHKPT, self.cutoff_layer, 3 * FRAMES, "video layers",no_class=no_class, drop=pt_attn_drop)
        self.multimodal_video = clones(Block(embed_dim, num_head, attn_drop=attn_drop, qk_norm=True), self.num_multimodal_layers)
        self.vcls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.hidden_sz = 512
        self.head = nn.Sequential(
            nn.Linear(self.num_modalities * embed_dim, self.hidden_sz), # there are 2 modalities, each with embed_dim features
            nn.Dropout(linear_drop),
            nn.Linear(self.hidden_sz, self.num_class)
        ) 
        self.sigmoid = nn.Sigmoid()
        self.video_augmentations = nn.Sequential(
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(),
        ) if apply_augmentation else nn.Identity()

        assert freeze_first <= self.cutoff_layer, f"freeze_first must be at most the number of layers until the cutoff layer: {self.cutoff_layer}"
        for layer_num in range(freeze_first):
            self.unimodal_video.model.blocks[layer_num].requires_grad_(False)


        reset_weights(self.unimodal_video.model.blocks, freeze_first)

    def forward(self, v):
        B = v.shape[0] # Batch size
        v = self.unimodal_video.model.embed_project(v)
        v = self.unimodal_video.model.forward_features(v)

        for i in range(self.num_multimodal_layers):
            v = self.multimodal_video[i](v)

        out = v[:, :1, :] 
        out = out.flatten(start_dim=1, end_dim=2)
        out = self.head(out)
        out = out.reshape(B, 1, self.num_class)
        out = self.sigmoid(out)
        return out

def train_video(net, trainldr, optimizer, loss_fn, cls_metrics):
    total_losses = AverageMeter()
    net.train()
    cls_metrics.reset()
    for data in tqdm(trainldr):
        _, video, labels = data
        batch_sz = video.shape[0]
        video = video.reshape(batch_sz * 10, 3, 224, 224)
        video = net.module.video_augmentations(video)
        video = video.reshape(batch_sz, 30, 224, 224)
        video = video.to(DEVICE)
        labels = labels.float()
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        y = net(video)
        loss = loss_fn(y, labels)
        y = (y > net.module.classification_threshold).float()
        cls_metrics.update(y.squeeze(1), labels.squeeze(1))
        loss.backward()
        optimizer.step()
        total_losses.update(loss.data.item(), labels.size(0))

    cls_metrics.avg()
    return total_losses.avg(), cls_metrics

def val_video(net, valldr, loss_fn, cls_metrics):
    total_losses = AverageMeter()
    cls_metrics.reset()
    net.eval()
    with torch.no_grad():
        for data in tqdm(valldr):
            _, video, labels = data
            video = video.to(DEVICE)
            labels = labels.float()
            labels = labels.to(DEVICE)

            y = net(video)
            loss = loss_fn(y, labels)

            y = (y > net.module.classification_threshold).float()
            cls_metrics.update(y.squeeze(1), labels.squeeze(1))
            total_losses.update(loss.data.item(), labels.size(0))
    cls_metrics.avg()
    return total_losses.avg(), cls_metrics



if __name__ == "__main__":
    EPOCHS = 200
    lr = 0.001
    BATCH_SZ = 8
    LABELS = 1
    vmbt = ViTMBT(1024, num_class=LABELS)
    vmbt = nn.DataParallel(vmbt).cuda()
    train_ds = EmoDataset(f"{DATA_DIR}/Labels/train_labels_full_pruned.csv", nlines=None, sole_emotion=None) # emotion #4 scores very high
    val_ds = EmoDataset(f"{DATA_DIR}/Labels/val_labels_full_pruned.csv", nlines=50, sole_emotion=None)
    test_ds = EmoDataset(f"{DATA_DIR}/Labels/test_labels_full_pruned.csv", nlines=None, sole_emotion=None)

    train_metrics = ClassifierMetrics(task='binary', n_labels=LABELS, device=DEVICE)
    val_metrics = ClassifierMetrics(task='binary', n_labels=LABELS, device=DEVICE)

    train_dl = DataLoader(train_ds, collate_fn=new_collate_fn, batch_size=BATCH_SZ, shuffle=True)
    val_dl = DataLoader(val_ds, collate_fn=new_collate_fn, batch_size=BATCH_SZ, shuffle=False)
    test_dl = DataLoader(test_ds, collate_fn=new_collate_fn, batch_size=BATCH_SZ, shuffle=False)

    optimizer = torch.optim.AdamW(vmbt.parameters(), betas=(0.9, 0.999), lr=lr, weight_decay=1.0 / BATCH_SZ)
    # loss_func = BCELoss()
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        train_metrics = train(vmbt, train_dl, optimizer, loss_fn=loss_func, cls_metrics=train_metrics)
        train_loss, train_f1, train_r, train_p, train_acc, train_bottleneck_tokens = train_metrics
        val_metrics = val(vmbt, val_dl, loss_fn=loss_func, cls_metrics=val_metrics)
        val_loss, val_f1, val_r, val_p, val_acc, val_bottleneck_tokens = val_metrics
        train_token = train_bottleneck_tokens[0][:, :256]
        val_token = val_bottleneck_tokens[0][:, :256]

        print(
            (f"Epoch {epoch + 1}: train_loss {train_loss:.5f}, val_loss {val_loss:.5f}\n"
                f"                   train_precision {train_p}, val_precision {val_p}\n"
                f"                   train_recall {train_r}, val_recall {val_r}\n"
                f"                   train_f1 {train_f1}, val_f1 {val_f1}"
                )
            )
    



    exit()
    pvt = PretrainedViT(PRETRAINED_CHKPT, 8, "audio layers")

    img = Image.open("dog.jpg").convert('RGB')

    config = {
        'input_size': (3, 224, 224), 
        'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 
        'std': (0.5, 0.5, 0.5),
        'crop_pct': 0.9, 'crop_mode': 'center'
    }
    transform = create_transform(**config)
    tensor = transform(img).unsqueeze(0)  # transform and add batch dimension

    with torch.no_grad():
        out = pvt(tensor)
        print(out.shape)
    probabilities = torch.nn.functional.softmax(out[0], dim=0)
    print(probabilities.shape)
    # prints: torch.Size([1000])

    # Get imagenet class mappings
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]

    # Print top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    for i in range(top5_prob.size(0)):
        print(categories[top5_catid[i]], top5_prob[i].item())