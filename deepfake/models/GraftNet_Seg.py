import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from deepfake.models.F3Net.attention import SimplifiedScaledDotProductAttention
from deepfake.models.F3Net.models import *
from deepfake.models.F3Net.xception import Block


class Decoder(nn.Module):
    def __init__(self, in_channel=32) -> None:
        super().__init__()
        self.upconv1 = nn.ConvTranspose2d(in_channel, 16, 2, 2, 0) #[N,32,28,28] => [N,16,56,56]
        self.fusion1 = nn.Sequential(                     #[N,32,56,56] => [N,16,56,56]
            nn.Conv2d(32, 16, 1),
            nn.Conv2d(16, 16, 3, 1, 1)
        )       
        self.upconv2 = nn.ConvTranspose2d(16, 8, 2, 2, 0) #[N,16,56,56] => [N,8,112,112]
        self.fusion2 = nn.Sequential(                     #[N,16,112,112] => [N,8,112,112]
            nn.Conv2d(16, 8, 1),
            nn.Conv2d(8, 8, 3, 1, 1)
        )
        self.upconv3 = nn.ConvTranspose2d(8, 4, 2, 2, 0)  #[N,8,112,112] => [N,4,224,224]
        self.fusion3 = nn.Sequential(                     #[N,8,224,224] => [N,4,224,224]
            nn.Conv2d(8, 4, 1),
            nn.Conv2d(4, 4, 3, 1, 1)
        )
        self.final   = nn.Conv2d(4, 1, 1)                 #[N,4,224,224] => [N,1,224,224]
        self.sigmod  = nn.Sigmoid()

        self.FAD_cvt = nn.Conv2d(32, 16, 1)        #[N,32,56,56] => [N,16,56,56]
        self.LFS_cvt = nn.Conv2d(6, 8, 1)          #[N,6,112,112] => [N,8,112,112]
        self.IMG_cvt = nn.Conv2d(3, 4, 1)          #[N,3,224,224] => [N,4,224,224]


    def forward(self, input, ds_feas:list):
        feas = self.upconv1(input)
        seg_FAD = self.FAD_cvt(ds_feas[-1])
        feas = torch.cat((seg_FAD, feas), dim=1)    #concat v->FAD
        feas = self.fusion1(feas)

        feas = self.upconv2(feas)
        seg_LFS = self.LFS_cvt(ds_feas[-2])
        feas = torch.cat((seg_LFS, feas), dim=1)    #concat v->LFS
        feas = self.fusion2(feas)

        feas = self.upconv3(feas)
        seg_IMG = self.IMG_cvt(ds_feas[-3])
        feas = torch.cat((seg_IMG, feas), dim=1)    #concat v->IMG
        feas = self.fusion3(feas)

        output = self.final(feas)
        output = self.sigmod(output)
        return output


class FAD_conv(nn.Module):
    def __init__(self, in_channel=12):
        super().__init__()
        self.preconv = nn.Sequential(
            nn.Conv2d(in_channel, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.XBlock1 = Block(32,32,2,2)     
        self.XBlock2 = Block(32,32,2,2)     

    def forward(self,input):
        feas = self.preconv(input)         #[bs,32,112,112]
        seg_FAD = self.XBlock1(feas)       #[bs,32,56,56]
        out = self.XBlock2(seg_FAD)        #[bs,32,28,28]
        return seg_FAD, out


class LFS_conv(nn.Module):
    def __init__(self, in_channel=6):
        super().__init__()
        self.XBlock1 = Block(in_channel,32,2,2)     
        self.XBlock2 = Block(32,32,2,2)  

    def forward(self, input):
        seg_LFS = input
        feas = self.XBlock1(input)          #[bs,32,56,56]
        feas = self.XBlock2(feas)           #[bs,32,28,28]
        return seg_LFS, feas


class GraftNet_Seg(nn.Module):
    def __init__(self, num_classes=3, img_width=224, img_height=224, feas_shape=28*28, head_layers=8, LFS_window_size=10, LFS_M = 6, writer=SummaryWriter()):
        super(GraftNet_Seg, self).__init__()
        assert img_width == img_height
        self.img_width = img_width
        self.img_height = img_height
        self.num_classes = num_classes
        self.window_size = LFS_window_size
        self._LFS_M = LFS_M
        self.writer = writer

        self.FAD_head = FAD_Head(self.img_width)
        self.LFS_head = LFS_Head(self.img_width, LFS_window_size, LFS_M)
        self.FAD_conv = FAD_conv(in_channel=12)
        self.LFS_conv = LFS_conv(in_channel=6)
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

        self.decoder = Decoder(in_channel=32)

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
        seg_IMG = input

        fea_FAD = self.FAD_head(input)      #[bs, 12, 224, 224]
        seg_FAD, fea_FAD = self.FAD_conv(fea_FAD)    #[bs, 32, 28, 28]
        fea_FAD = self.FAD_project(fea_FAD.view(self.project_reshape(fea_FAD)))

        fea_LFS = self.LFS_head(input)      #[bs, 6, 112, 112]
        seg_LFS, fea_LFS = self.LFS_conv(fea_LFS)    #[bs, 32, 28, 28]
        fea_LFS = self.LFS_project(fea_LFS.view(self.project_reshape(fea_LFS)))

        fea_img = self.IMG_head(input)      #[bs, 32, 28, 28]
        fea_img = self.IMG_project(fea_img.view(self.project_reshape(fea_img)))

        fea_att = self.attention(fea_img, fea_LFS, fea_FAD)     #[bs, 32, 784]
        fea_att = fea_att.view(self.project_reshape(fea_att, mode='split'))     #[bs, 32, 28, 28]
        # print(fea_FAD.shape, fea_LFS.shape, fea_img.shape, fea_att.shape)

        # seg
        ds_feas = [seg_IMG, seg_LFS, seg_FAD]
        seg = self.decoder(fea_att, ds_feas)

        # cls
        cls = self.cls(fea_att)             #[bs, 256, 7, 7]
        cls = self._norm_fea(cls)           #[bs, 256]
        out = self.dp(cls)
        out = self.fc(out)
        return seg, out


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
    model = GraftNet_Seg(writer=None)
    seg, out = model(input)
    print(seg.shape, out.shape)
    total_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(total_num)

