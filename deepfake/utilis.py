import torch
import torch.nn as nn
import torchvision.models as models

from vit_pytorch import ViT
from deepfake.models.F3Net.GraftNet import GraftNet
from deepfake.models.F3Net.GraftNet_Seg import GraftNet_Seg
from deepfake.models.F3Net.models import F3Net
from deepfake.models.F3Net.F3Net_Seg import F3Net_Seg 


def init_model(cfg, mode='train', type='resnet', path=None, class_nums=3, writer=None):
    if mode == 'train':
        if type == 'resnet':
            model = models.resnet18()
            print("Loading the resnet model.")
            if path is not None:
                model.load_state_dict(torch.load(path))
                print("Complete loading the resnet pretrained model.")
            model.fc = nn.Linear(in_features=model.fc.in_features, out_features=class_nums)
        elif type == 'ViT':
            model = ViT(
                image_size=224,
                patch_size=14,
                num_classes=class_nums,
                dim=512,
                depth = 6,
                heads=6,
                mlp_dim=512,
                dropout=0.1,
                emb_dropout=0.1
            )
            print("Complete loading the vit model.")
        elif type == 'F3Net':
            model = F3Net(
                num_classes=3,
                img_width=cfg.img_size[0],
                img_height=cfg.img_size[1],
                mode='Original',
                pretrain=path
            )
            print("Loading the F3Net model.")
        elif type == 'F3Net_Seg':
            model = F3Net_Seg(
                num_classes=3,
                img_width=cfg.img_size[0],
                img_height=cfg.img_size[1],
                mode='Multi_task',
                pretrain=path,
                writer=writer
            )
        elif type == 'GraftNet':
            model = GraftNet(          
                num_classes=3,
                img_width=cfg.img_size[0],
                img_height=cfg.img_size[1],
                feas_shape=28*28,
                head_layers=8, 
                LFS_window_size=10, 
                LFS_M=6,
                writer=writer
            )
        elif type == 'GraftNet_Seg':
            model = GraftNet_Seg(          
                num_classes=3,
                img_width=cfg.img_size[0],
                img_height=cfg.img_size[1],
                feas_shape=28*28,
                head_layers=8, 
                LFS_window_size=10, 
                LFS_M=6,
                writer=writer
            )

    # predict
    else:
        if type == 'resnet':
            model = models.resnet18()
            model.fc = nn.Linear(in_features=model.fc.in_features, out_features=class_nums)
            if cfg.multi_gpu:
                model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(path).items()})
            else:
                model.load_state_dict(torch.load(path))
            print("Complete loading the resnet model. Start evaluation...")
        elif type == 'F3Net':
            model = F3Net(
                num_classes=3,
                img_width=cfg.img_size[0],
                img_height=cfg.img_size[1],
                mode='Both',
                pretrain=None
            )
            if cfg.multi_gpu:
                model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(path).items()})
            else:
                model.load_state_dict(torch.load(path))
            print("Complete loading the F3Net model. Start evaluation...")
        elif type == 'F3Net_Seg':
            model = F3Net_Seg(
                num_classes=3,
                img_width=cfg.img_size[0],
                img_height=cfg.img_size[1],
                mode='Multi_task',
                pretrain=path
            )
            if cfg.multi_gpu:
                model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(path).items()})
            else:
                model.load_state_dict(torch.load(path))
            print("Complete loading the Multi_task F3Net model. Start evaluation...")
        elif type == 'GraftNet':
            model = GraftNet(          
                num_classes=3,
                img_width=cfg.img_size[0],
                img_height=cfg.img_size[1],
                feas_shape=28*28,
                head_layers=8, 
                LFS_window_size=10, 
                LFS_M=6,
                writer=writer
            )
            if cfg.multi_gpu:
                model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(path).items()})
            else:
                model.load_state_dict(torch.load(path))
            print("Complete loading the GraftNet model. Start evaluation...")

    print("--------------------------------------------------------------------------")
    print(model)
    print("--------------------------------------------------------------------------")
    return model


def dice_loss(input, target):
    smooth = 1e-4
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))

def dice_score(pred, label):
    smooth = 1e-4
    iflat = label.view(-1)
    tflat = pred.view(-1)
    intersection = (iflat * tflat).sum()
    return ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))
