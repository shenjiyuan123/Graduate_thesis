import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from deepfake.models.F3Net.attention import SimplifiedScaledDotProductAttention
from deepfake.models.F3Net.models import *
from deepfake.models.F3Net.xception import Block


class GraftNet(nn.Module):
    def __init__(self, num_classes=3, img_width=224, img_height=224, feas_shape=28*28, head_layers=8, LFS_window_size=10, LFS_M = 6, writer=SummaryWriter()):
        super(GraftNet, self).__init__()
        assert img_width == img_height
        self.img_width = img_width
        self.img_height = img_height
        self.num_classes = num_classes
        self.window_size = LFS_window_size
        self._LFS_M = LFS_M
        self.writer = writer

        self.FAD_head = FAD_Head(self.img_width)
        self.LFS_head = LFS_Head(self.img_width, LFS_window_size, LFS_M)
        self.FAD_conv = nn.Sequential(
            nn.Conv2d(12, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            Block(32,32,2,2),
            Block(32,32,2,2)
        )
        self.LFS_conv = nn.Sequential(
            Block(6,32,2,2),
            Block(32,32,2,2)
        )
        self.IMG_head =nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            Block(16,32,2,2),
            Block(32,32,2,2)
        )
        self.FAD_project = nn.Linear(feas_shape, feas_shape)
        self.LFS_project = nn.Linear(feas_shape, feas_shape)
        self.IMG_project = nn.Linear(feas_shape, feas_shape)
        self.attention = SimplifiedScaledDotProductAttention(d_model=feas_shape, h=head_layers)

        self.cls = nn.Sequential(
            Block(32,64,2,2),
            Block(64,128,2,2),
            Block(128,256,3,1),
            Block(256,256,3,1)
        )
        self.dp = nn.Dropout(p=0.2)
        self.fc = nn.Linear(256, num_classes)

    def _norm_fea(self, fea):
        f = F.relu(fea, inplace=True)
        f = F.adaptive_avg_pool2d(f,(1,1))
        f = f.view(f.size(0), -1)
        return f

    def forward(self, input):
        fea_FAD = self.FAD_head(input)      #[bs, 12, 224, 224]
        fea_FAD = self.FAD_conv(fea_FAD)    #[bs, 32, 28, 28]
        fea_FAD = self.FAD_project(fea_FAD.view(self.project_reshape(fea_FAD)))

        fea_LFS = self.LFS_head(input)      #[bs, 6, 112, 112]
        fea_LFS = self.LFS_conv(fea_LFS)    #[bs, 32, 28, 28]
        fea_LFS = self.LFS_project(fea_LFS.view(self.project_reshape(fea_LFS)))

        fea_img = self.IMG_head(input)      #[bs, 32, 28, 28]
        fea_img = self.IMG_project(fea_img.view(self.project_reshape(fea_img)))

        fea_att = self.attention(fea_img, fea_LFS, fea_FAD)     #[bs, 32, 784]
        fea_att = fea_att.view(self.project_reshape(fea_att, mode='split'))     #[bs, 32, 28, 28]
        # print(fea_FAD.shape, fea_LFS.shape, fea_img.shape, fea_att.shape)
        self.writer.add_image('fea_att/1', fea_att[0,0,:,:], dataformats='HW')
        self.writer.add_image('fea_att/2', fea_att[0,3,:,:], dataformats='HW')
        self.writer.add_image('fea_att/3', fea_att[0,6,:,:], dataformats='HW')

        cls = self.cls(fea_att)             #[bs, 256, 7, 7]
        cls = self._norm_fea(cls)           #[bs, 256]
        out = self.dp(cls)
        out = self.fc(out)
        return out


    def project_reshape(self, x, mode='integrate'):
        if mode=='integrate':
            bs, c, h, w = x.size()
            return (bs, c, h*w)
        elif mode=='split':
            bs, c, hw = x.size()
            assert hw == 28*28
            return (bs, c, 28, 28)


if __name__=="__main__":
    input = torch.randn(64,3,224,224)
    model = GraftNet()
    print(type(model(input)))
    total_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(total_num)

