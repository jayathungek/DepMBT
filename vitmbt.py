import time
import copy
import logging
import dataclasses
from tqdm import tqdm
from typing import Optional, Dict, Callable

from PIL import Image
import torch
import torch.nn as nn
from torch.nn import BCELoss
import torch.nn.functional as F
from torch.utils.data import DataLoader
from timm.models._manipulate import adapt_input_conv
from timm.models._helpers import load_state_dict
from timm.data.transforms_factory import create_transform
from timm.models._pretrained import PretrainedCfg
from timm.models._registry import get_pretrained_cfg
from timm.data import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

from helpers import *
from data import EmoDataset, new_collate_fn, DATA_DIR, WIDTH, HEIGHT, FRAMES
from vision_transformer import VisionTransformer, Block



DEVICE = "cuda"
PRETRAINED_CHKPT = "./pretrained_models/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz"

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


def resolve_pretrained_cfg(
        variant: str,
        pretrained_cfg=None,
        pretrained_cfg_overlay=None,
) -> PretrainedCfg:
    model_with_tag = variant
    pretrained_tag = None
    if pretrained_cfg:
        if isinstance(pretrained_cfg, dict):
            # pretrained_cfg dict passed as arg, validate by converting to PretrainedCfg
            pretrained_cfg = PretrainedCfg(**pretrained_cfg)
        elif isinstance(pretrained_cfg, str):
            pretrained_tag = pretrained_cfg
            pretrained_cfg = None

    # fallback to looking up pretrained cfg in model registry by variant identifier
    if not pretrained_cfg:
        if pretrained_tag:
            model_with_tag = '.'.join([variant, pretrained_tag])
        pretrained_cfg = get_pretrained_cfg(model_with_tag)

    if not pretrained_cfg:
        _logger.warning(
            f"No pretrained configuration specified for {model_with_tag} model. Using a default."
            f" Please add a config to the model pretrained_cfg registry or pass explicitly.")
        pretrained_cfg = PretrainedCfg()  # instance with defaults

    pretrained_cfg_overlay = pretrained_cfg_overlay or {}
    if not pretrained_cfg.architecture:
        pretrained_cfg_overlay.setdefault('architecture', variant)
    pretrained_cfg = dataclasses.replace(pretrained_cfg, **pretrained_cfg_overlay)

    return pretrained_cfg


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




class PretrainedViT(nn.Module):
    def __init__(self, pretrained_checkpoint_path, cutoff_layer, channels, model_name):
        super().__init__()
        print(f"Loading {model_name}...")
        t_start = time.time()
        cfg = _cfg(pretrained_checkpoint_path)
        self.model = VisionTransformer(in_chans=channels)
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
        


class ViTMBT(nn.Module):
    def __init__(self, embed_dim, num_bottle_token=4, bottle_layer=7
                , project_type='conv2d', num_head=4, drop=.1, num_layers=12, num_class=8):
        super().__init__()
        self.num_class = num_class
        self.num_layers = num_layers
        self.bottle_layer = bottle_layer
        self.num_bottle_token = num_bottle_token
        self.num_multimodal_layers = num_layers - bottle_layer
        # make vision transformer layers be accessible via subscript
        self.unimodal_audio = PretrainedViT(PRETRAINED_CHKPT, 8, 3, "audio layers")
        self.unimodal_video = PretrainedViT(PRETRAINED_CHKPT, 8, 3 * FRAMES, "video layers")
        self.multimodal_audio = clones(Block(embed_dim, num_head), self.num_multimodal_layers)
        self.multimodal_video = clones(Block(embed_dim, num_head), self.num_multimodal_layers)
        self.bottleneck_token = nn.Parameter(torch.zeros(1, num_bottle_token, embed_dim))
        self.acls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.vcls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.head = nn.Linear(2 * embed_dim, num_class) # there are 2 modalities, each with embed_dim features
        self.softmax = nn.Softmax()


    def forward(self, a, v):
        B = a.shape[0] # Batch size
        a = self.unimodal_audio.model.embed_project(a)
        v = self.unimodal_video.model.embed_project(v)

        acls_tokens = self.acls_token.expand(B, -1, -1)
        a = torch.cat((acls_tokens, a), dim=1)
        
        vcls_tokens = self.vcls_token.expand(B, -1, -1)
        v = torch.cat((vcls_tokens, v), dim=1)

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
        out = out.reshape(B, self.num_class)
        out = self.softmax(out)
        return out

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




def train(net, trainldr, optimizer, loss_fn, cls_metrics):
    total_losses = AverageMeter()
    net.train()
    for batch_idx, data in enumerate(tqdm(trainldr)):
        audio, video, labels = data
        audio = audio.to(DEVICE)
        video = video.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()

        # y = net(feature_audio, feature_video)
        y = net(audio, video)
        loss = loss_fn(y, labels)
        loss.backward()
        optimizer.step()

        # cls_metrics.calc(y, labels)
        cls_metrics.update(y, labels)
        total_losses.update(loss.data.item(), audio.size(0))

    cls_metrics.avg()
    return total_losses.avg()

def val(net, valldr, loss_fn, cls_metrics):
    total_losses = AverageMeter()
    net.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(valldr)):
            audio, video, labels = data
            audio = audio.to(DEVICE)
            video = video.to(DEVICE)
            labels = labels.to(DEVICE)

            # y = net(feature_audio, feature_video)
            y = net(audio, video)
            loss = loss_fn(y, labels)

            # cls_metrics.calc(y, labels)
            cls_metrics.update(y, labels)
            total_losses.update(loss.data.item(), audio.size(0))
    cls_metrics.avg()
    return total_losses.avg()

            
if __name__ == "__main__":
    EPOCHS = 200
    lr = 0.01
    batch_sz = 1
    LABELS = 8
    vmbt = ViTMBT(1024, num_class=LABELS)
    vmbt = vmbt.to(DEVICE)
    train_ds = EmoDataset(f"{DATA_DIR}/Labels/train_labels_less.txt", nlines=None)
    val_ds = EmoDataset(f"{DATA_DIR}/Labels/val_labels_full.txt", nlines=5)
    test_ds = EmoDataset(f"{DATA_DIR}/Labels/test_labels_full.txt", nlines=5)

    train_metrics = ClassifierMetrics(n_labels=LABELS, device=DEVICE)
    val_metrics = ClassifierMetrics(n_labels=LABELS, device=DEVICE)

    train_dl = DataLoader(train_ds, collate_fn=new_collate_fn, batch_size=batch_sz, shuffle=True)
    val_dl = DataLoader(val_ds, collate_fn=new_collate_fn, batch_size=batch_sz, shuffle=False)
    test_dl = DataLoader(test_ds, collate_fn=new_collate_fn, batch_size=batch_sz, shuffle=False)

    optimizer = torch.optim.AdamW(vmbt.parameters(), betas=(0.9, 0.999), lr=lr, weight_decay=1.0 / batch_sz)
    loss_func = BCELoss()
    for epoch in range(EPOCHS):
        train_loss = train(vmbt, train_dl, optimizer, loss_fn=loss_func, cls_metrics=train_metrics)
        val_loss = val(vmbt, val_dl, loss_fn=loss_func, cls_metrics=val_metrics)
        print(
            (f"Epoch {epoch + 1}: train_loss {train_loss:.5f}, val_loss {val_loss:.5f}\n"
             f"                   train_AP {train_metrics.ap}, val_AP {val_metrics.ap}\n"
             f"                   train_precision {train_metrics.precision}, val_AP {val_metrics.precision}\n"
             f"                   train_recall {train_metrics.recall}, val_recall {val_metrics.recall}\n"
             f"                   train_f1 {train_metrics.f1}, val_f1 {val_metrics.f1}"
             )
            )
        train_metrics.reset()
        val_metrics.reset()



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