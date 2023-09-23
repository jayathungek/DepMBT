# Confirm unimodal w/ just audio works

transforms.RandomRotation(degrees=15),
transforms.RandomHorizontalFlip(p=0.2),
transforms.RandomInvert(p=0.2)
## Multimodal 
(class_thresh: 0.75, hidden_layer=512, 0.1 linear dropout, 0.1 attn dropout, cutoff_layer=20, freeze_first=18, num_layers=24, apply_augmentation=False)
SPLIT = [0.9, 0.05, 0.05]
{'batch_sz': 32,
 'epochs': 15,
 'loss': 'BCELoss',
 'optim': 'AdamW',
 'optim_lr': 1e-05,
 'optim_momentum': None,
 'optim_weight_decay': 0.03125,
 'train': {'acc': 0.9508064389228821,
           'f1': 0.7619515657424927,
           'loss': 0.09227314864851764,
           'precision': 0.8770595192909241,
           'recall': 0.7004812359809875},
 'val': {'acc': 0.7601207494735718,
         'f1': 0.271915078163147,
         'loss': 0.7789999224521496,
         'precision': 0.3315476179122925,
         'recall': 0.25589826703071594}}

using new audio loading - padding instead of resizing
(class_thresh: 0.75, hidden_layer=512, 0.1 linear dropout, 0.1 attn dropout, cutoff_layer=20, freeze_first=18, num_layers=24, apply_augmentation=False)
SPLIT = [0.9, 0.05, 0.05]

{'batch_sz': 32,
 'epochs': 15,
 'loss': 'BCELoss',
 'optim': 'AdamW',
 'optim_lr': 1e-05,
 'optim_momentum': None,
 'optim_weight_decay': 0.03125,
 'train': {'acc': 0.9447832107543945,
           'f1': 0.7433571815490723,
           'loss': 0.10331586797520058,
           'precision': 0.8566735982894897,
           'recall': 0.6803011894226074},
 'val': {'acc': 0.8140980005264282,
         'f1': 0.3851699233055115,
         'loss': 0.6108025621484827,
         'precision': 0.5690972208976746,
         'recall': 0.30669304728507996}}

(class_thresh: 0.75, hidden_layer=512, 0.5 linear dropout, 0.5 attn dropout, cutoff_layer=20, freeze_first=18, num_layers=24, apply_augmentation=True)
split = [0.9, 0.05, 0.05]

## Unimodal - audio
(class_thresh: 0.75, hidden_layer=512, 0.1 linear dropout, 0.1 attn dropout, cutoff_layer=20, freeze_first=18, num_layers=24, apply_augmentation=False)
SPLIT = [0.9, 0.05, 0.05]
{'batch_sz': 32,
 'epochs': 15,
 'loss': 'BCELoss',
 'optim': 'AdamW',
 'optim_lr': 1e-05,
 'optim_momentum': None,
 'optim_weight_decay': 0.03125,
 'train': {'acc': 0.792137086391449,
           'f1': 0.2395651936531067,
           'loss': 0.3662730472604024,
           'precision': 0.507519006729126,
           'recall': 0.1696183830499649},
 'val': {'acc': 0.787109375,
         'f1': 0.33020928502082825,
         'loss': 0.4845331178771125,
         'precision': 0.4568108916282654,
         'recall': 0.27389174699783325}}

using new audio loading - padding instead of resizing
(class_thresh: 0.75, hidden_layer=512, 0.1 linear dropout, 0.1 attn dropout, cutoff_layer=20, freeze_first=18, num_layers=24, apply_augmentation=False)
SPLIT = [0.9, 0.05, 0.05]
{'batch_sz': 32,
 'epochs': 15,
 'loss': 'BCELoss',
 'optim': 'AdamW',
 'optim_lr': 1e-05,
 'optim_momentum': None,
 'optim_weight_decay': 0.03125,
 'train': {'acc': 0.8223286271095276,
           'f1': 0.3664609491825104,
           'loss': 0.3187472538849742,
           'precision': 0.6765802502632141,
           'recall': 0.2725372016429901},
 'val': {'acc': 0.7489346265792847,
         'f1': 0.2799678146839142,
         'loss': 0.5432118133262351,
         'precision': 0.4833333492279053,
         'recall': 0.2277800440788269}}



## Unimodal - video
(class_thresh: 0.75, hidden_layer=512, 0.1 linear dropout, 0.1 attn dropout, cutoff_layer=20, freeze_first=18, num_layers=24, apply_augmentation=False)
SPLIT = [0.9, 0.05, 0.05]
{'batch_sz': 32,
 'epochs': 15,
 'loss': 'BCELoss',
 'optim': 'AdamW',
 'optim_lr': 1e-05,
 'optim_momentum': None,
 'optim_weight_decay': 0.03125,
 'train': {'acc': 0.9081653356552124,
           'f1': 0.6383172869682312,
           'loss': 0.16640998395447879,
           'precision': 0.8316358923912048,
           'recall': 0.5483762621879578},
 'val': {'acc': 0.8014914989471436,
         'f1': 0.3567032217979431,
         'loss': 0.5639171754872357,
         'precision': 0.6050595045089722,
         'recall': 0.27470237016677856}}

# Vision transformer is bad for video, need to change backbone
this could be the main contribution of the work - vision transformer adapted for video, 
specifically for emotion detection 