# Confirm unimodal w/ just audio works
(class_thresh: 0.75, hidden_layer=512, 0.3 linear dropout, 0.4 attn dropout, cutoff_layer=20, freeze_first=18, num_layers=24, apply_augmentation=False)
SPLIT = [0.95, 0.05, 0]
Augmentations (for video only)
transforms.RandomRotation(degrees=15),
transforms.RandomHorizontalFlip(),
transforms.RandomInvert(),
transforms.ColorJitter(),
transforms.RandomCrop((224, 224)),
transforms.RandomAffine(degrees=10),
transforms.RandomAdjustSharpness(2)

## Multimodal 
{'batch_sz': 32,
 'epochs': 15,
 'loss': 'BCELoss',
 'optim': 'AdamW',
 'optim_lr': 1e-05,
 'optim_momentum': None,
 'optim_weight_decay': 0.03125,
 'train': {'acc': 0.8887852430343628,
           'f1': 0.5820860266685486,
           'loss': 0.21493807044840352,
           'precision': 0.7861285209655762,
           'recall': 0.4950549602508545},
 'val': {'acc': 0.7576349377632141,
         'f1': 0.3853834569454193,
         'loss': 0.6265007169158371,
         'precision': 0.5747023820877075,
         'recall': 0.32969164848327637}}


## Unimodal - audio
{'batch_sz': 32,
 'epochs': 15,
 'loss': 'BCELoss',
 'optim': 'AdamW',
 'optim_lr': 1e-05,
 'optim_momentum': None,
 'optim_weight_decay': 0.03125,
 'train': {'acc': 0.7390876412391663,
           'f1': 0.05745992809534073,
           'loss': 0.608884160911914,
           'precision': 0.14443151652812958,
           'recall': 0.03889245167374611},
 'val': {'acc': 0.6977983117103577,
         'f1': 0.0,
         'loss': 0.5844192769792345,
         'precision': 0.0,
         'recall': 0.0}}


## Unimodal - video
{'batch_sz': 32,
 'epochs': 15,
 'loss': 'BCELoss',
 'optim': 'AdamW',
 'optim_lr': 1e-05,
 'optim_momentum': None,
 'optim_weight_decay': 0.03125,
 'train': {'acc': 0.7448840141296387,
           'f1': 0.04143529385328293,
           'loss': 0.5615926930584858,
           'precision': 0.09505043923854828,
           'recall': 0.029804643243551254},
 'val': {'acc': 0.7386363744735718,
         'f1': 0.06587009876966476,
         'loss': 0.5474473856113575,
         'precision': 0.06587009876966476,
         'recall': 0.06587009876966476}}

# Vision transformer is bad for video, need to change backbone
this could be the main contribution of the work - vision transformer adapted for video, 
specifically for emotion detection 