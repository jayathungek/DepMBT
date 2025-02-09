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

using new audio loading - padding instead of resizing
(class_thresh: 0.75, hidden_layer=512, 0.5 linear dropout, 0.5 attn dropout, cutoff_layer=20, freeze_first=18, num_layers=24, apply_augmentation=True)
split = [0.9, 0.05, 0.05]
{'batch_sz': 32,
 'epochs': 25,
 'loss': 'BCELoss',
 'optim': 'AdamW',
 'optim_lr': 1e-05,
 'optim_momentum': None,
 'optim_weight_decay': 0.03125,
 'train': {'acc': 0.8744707107543945,
           'f1': 0.5516762137413025,
           'loss': 0.22906572486936433,
           'precision': 0.7741323113441467,
           'recall': 0.462685227394104},
 'val': {'acc': 0.798828125,
         'f1': 0.3314495086669922,
         'loss': 0.5450333544501552,
         'precision': 0.4451252222061157,
         'recall': 0.304807186126709}}

using new audio loading - padding instead of resizing, Cosine annealing LR
(class_thresh: 0.75, hidden_layer=512, 0.5 linear dropout, 0.5 attn dropout, cutoff_layer=20, freeze_first=18, num_layers=24, apply_augmentation=True)
split = [0.9, 0.05, 0.05]
{'batch_sz': 32,
 'best_epoch': 17,
 'epochs': 30,
 'loss': 'BCELoss',
 'optim': 'AdamW',
 'optim_lr': 5e-05,
 'optim_momentum': None,
 'optim_weight_decay': 0.4,
 'T_0': 26,
 'train': {'acc': 0.9849545955657959,
           'f1': 0.9109511375427246,
           'loss': 0.03241169482623179,
           'precision': 0.9384472370147705,
           'recall': 0.8929294347763062},
 'val': {'acc': 0.7311789989471436,
         'f1': 0.35093724727630615,
         'loss': 1.1934142134807728,
         'precision': 0.4828125238418579,
         'recall': 0.29820191860198975}}


using new audio loading - padding instead of resizing, Cosine annealing LR, dropout only applied to fusion and linear layers  
(class_thresh: 0.75, hidden_layer=512, 0.4 linear dropout, 0.4 attn dropout, cutoff_layer=20, freeze_first=18, num_layers=24, apply_augmentation=True)
split = [0.9, 0.05, 0.05]
{'batch_sz': 32,
 'best_epoch': 31,
 'epochs': 55,
 'loss': 'BCELoss',
 'optim': 'AdamW',
 'optim_lr': 5e-05,
 'optim_momentum': None,
 'optim_weight_decay': 0.4,
 'T_0': 6,
 'train': {'acc': 0.9967237710952759,
           'f1': 0.9452287554740906,
           'loss': 0.009313095987476792,
           'precision': 0.9507728219032288,
           'recall': 0.941108226776123},
 'val': {'acc': 0.7544388771057129,
         'f1': 0.4274536967277527,
         'loss': 1.6637600439566154,
         'precision': 0.5294011831283569,
         'recall': 0.4185143709182739}}

using new audio loading - padding instead of resizing, Cosine annealing LR, dropout applied to pretrained, linear and fusion layers
(class_thresh: 0.75, hidden_layer=512, cutoff_layer=20, freeze_first=18, num_layers=24, apply_augmentation=True)
split = [0.95, 0.05, 0.0]
{'T_0': 6,
 'batch_sz': 32,
 'best_epoch': 33,
 'dropout': 0.6,
 'epochs': 35,
 'loss': 'BCELoss',
 'optim': 'AdamW',
 'optim_lr': 5e-05,
 'optim_momentum': None,
 'optim_weight_decay': 0.4,
 'train': {'acc': 0.8260498046875,
           'f1': 0.4346005618572235,
           'loss': 0.3374116774648428,
           'precision': 0.6985670924186707,
           'recall': 0.3401227295398712},
 'val': {'acc': 0.7837358117103577,
         'f1': 0.30548566579818726,
         'loss': 0.5310498431876853,
         'precision': 0.44162946939468384,
         'recall': 0.24488280713558197}}

using new audio loading - padding instead of resizing, Cosine annealing LR, dropout applied to pretrained, linear and fusion layers
(class_thresh: 0.75, hidden_layer=512, cutoff_layer=20, freeze_first=18, num_layers=24, apply_augmentation=True)
split = [0.95, 0.05, 0.0]
{'T_0': 6,
 'batch_sz': 32,
 'best_epoch': 16,
 'dropout': 0.3,
 'epochs': 35,
 'loss': 'BCELoss',
 'optim': 'AdamW',
 'optim_lr': 5e-05,
 'optim_momentum': None,
 'optim_weight_decay': 0.4,
 'train': {'acc': 0.884521484375,
           'f1': 0.6278195381164551,
           'loss': 0.2129565179347992,
           'precision': 0.8437825441360474,
           'recall': 0.5259432196617126},
 'val': {'acc': 0.7606533765792847,
         'f1': 0.3125903606414795,
         'loss': 0.6558366660718564,
         'precision': 0.41364261507987976,
         'recall': 0.27420926094055176}}

using new audio loading - padding instead of resizing, Cosine annealing LR, dropout applied to linear and fusion layers
(class_thresh: 0.75, hidden_layer=512, cutoff_layer=20, freeze_first=18, num_layers=24, apply_augmentation=True)
split = [0.95, 0.05, 0.0]
{'T_0': 6,
 'attn_dropout': 0.8,
 'batch_sz': 32,
 'best_epoch': 23,
 'epochs': 35,
 'linear_dropout': 0.1,
 'loss': 'BCELoss',
 'optim': 'AdamW',
 'optim_lr': 5e-05,
 'optim_momentum': None,
 'optim_weight_decay': 0.4,
 'train': {'acc': 0.995361328125,
           'f1': 0.959815263748169,
           'loss': 0.01693048299057409,
           'precision': 0.97265625,
           'recall': 0.9507956504821777},
 'val': {'acc': 0.7633167505264282,
         'f1': 0.4201613664627075,
         'loss': 1.035628080368042,
         'precision': 0.5329365134239197,
         'recall': 0.36418652534484863}}

using new audio loading - padding instead of resizing, Cosine annealing LR, dropout applied to linear and fusion layers
0.01 dropout in pre-trained layers
(class_thresh: 0.75, hidden_layer=512, cutoff_layer=20, freeze_first=18, num_layers=24, apply_augmentation=True)
split = [0.95, 0.05, 0.0]
{'T_0': 6,
 'attn_dropout': 0.4,
 'batch_sz': 32,
 'best_epoch': 27,
 'epochs': 35,
 'linear_dropout': 0.1,
 'loss': 'BCELoss',
 'optim': 'AdamW',
 'optim_lr': 5e-05,
 'optim_momentum': None,
 'optim_weight_decay': 0.4,
 'train': {'acc': 0.9942626953125,
           'f1': 0.9629107713699341,
           'loss': 0.01260563558753347,
           'precision': 0.9709990620613098,
           'recall': 0.9567868113517761},
 'val': {'acc': 0.7375710010528564,
         'f1': 0.3817528784275055,
         'loss': 1.3595906893412273,
         'precision': 0.5141865015029907,
         'recall': 0.3408358097076416}}

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


using new audio loading - padding instead of resizing
(class_thresh: 0.75, hidden_layer=512, cutoff_layer=20, freeze_first=18, num_layers=24, apply_augmentation=False)
SPLIT = [0.95, 0.05, 0.0]
{'T_0': 6,
 'attn_dropout': 0.4,
 'batch_sz': 32,
 'best_epoch': 32,
 'epochs': 35,
 'linear_dropout': 0.1,
 'loss': 'BCELoss',
 'optim': 'AdamW',
 'optim_lr': 5e-05,
 'optim_momentum': None,
 'optim_weight_decay': 0.4,
 'pt_attn_dropout': 0.05,
 'train': {'acc': 0.989501953125,
           'f1': 0.9517146944999695,
           'loss': 0.02582615583378356,
           'precision': 0.9654729962348938,
           'recall': 0.9433674216270447},
 'val': {'acc': 0.7164417505264282,
         'f1': 0.2929023206233978,
         'loss': 1.683484156926473,
         'precision': 0.3301091194152832,
         'recall': 0.28019338846206665}}


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

(class_thresh: 0.75, hidden_layer=512, cutoff_layer=20, freeze_first=18, num_layers=24, apply_augmentation=True)
SPLIT = [0.95, 0.05, 0.0]
transforms.RandomRotation(degrees=15),
transforms.RandomHorizontalFlip(),
transforms.ColorJitter(),
# Vision transformer is bad for video, need to change backbone
this could be the main contribution of the work - vision transformer adapted for video, 
specifically for emotion detection 