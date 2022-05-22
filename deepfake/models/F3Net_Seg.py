import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import torchvision

from deepfake.models.F3Net.xception import Xception
from deepfake.models.F3Net.models import *
from torch.utils.tensorboard import SummaryWriter

class Decoder(nn.Module):
    def __init__(self, width, height) -> None:
        super().__init__()
        self.preconv = nn.Conv2d(728*2, 728, 1, 1, 0)
        self.upconv1 = nn.ConvTranspose2d(728, 256, 2, 2, 0) #[N,728,14,14] => [N,256,28,28]
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, 2, 0) #[N,256,28,28] => [N,128,56,56]
        self.upconv3 = nn.ConvTranspose2d(128, 64, 2, 2, 0)  #[N,128,56,56] => [N,64,112,112]
        self.upconv4 = nn.ConvTranspose2d(64, 32, 2, 2, 0)   #[N,64,112,112] => [N,32,224,224]
        self.final   = nn.Conv2d(32, 1, 1) #[N,32,224,224] => [N,1,224,224]
        self.sigmod  = nn.Sigmoid()

    def forward(self, input):
        feas = self.preconv(input)
        feas = self.upconv1(feas)
        feas = self.upconv2(feas)
        feas = self.upconv3(feas)
        feas = self.upconv4(feas)
        output = self.final(feas)
        output = self.sigmod(output)
        return output


class F3Net_Seg(nn.Module):
    def __init__(self, num_classes=3, img_width=224, img_height=224, LFS_window_size=10, LFS_stride=2, LFS_M = 6, mode='Multi_task', pretrain=None, writer=SummaryWriter()):
        super(F3Net_Seg, self).__init__()
        assert img_width == img_height
        self.img_width = img_width
        self.img_height = img_height
        self.num_classes = num_classes
        self.mode = mode
        self.window_size = LFS_window_size
        self._LFS_M = LFS_M
        self.pretrain = pretrain
        self.writer = writer

        if mode == 'Multi_task':
            self.FAD_head = FAD_Head(self.img_width)
            self.FAD_xcep = Xception(self.num_classes)
            self.FAD_xcep.conv1 = nn.Conv2d(12, 32, 3, 2, 0, bias=False)
            
            self.LFS_head = LFS_Head(self.img_width, LFS_window_size, LFS_M)
            self.LFS_xcep = Xception(self.num_classes)
            self.LFS_xcep.conv1 = nn.Conv2d(self._LFS_M, 32, 3, 1, 0, bias=False)
            
            self.Mixblock1 = MixBlock(728,14,14)
            self.Mixblock2 = MixBlock(728,14,14)

            self.Decoder = Decoder(img_width, img_height)

        # classifier
        # remember to add avg pool2d
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(4096, num_classes)
        self.dp = nn.Dropout(p=0.2)

    def _norm_fea(self, fea):
        f = self.relu(fea)
        f = F.adaptive_avg_pool2d(f, (1,1))
        f = f.view(f.size(0), -1)
        return f

    def forward(self, x):
        # Mixblock1
        fea_FAD = self.FAD_head(x)

        # visualization FAD
        tmp = fea_FAD[0].view((4,3,self.img_height,self.img_width))
        grid_fea_FAD = torchvision.utils.make_grid(tmp)
        self.writer.add_image('fea_FAD', grid_fea_FAD)
        
        fea_FAD = self.FAD_xcep.block_8_features(fea_FAD)
        fea_LFS = self.LFS_head(x)

        # visualization LFS
        tmp = fea_LFS[0].view((2,3,self.img_height//2,self.img_width//2))
        grid_fea_LFS = torchvision.utils.make_grid(tmp)
        self.writer.add_image('fea_LFS', grid_fea_LFS)

        fea_LFS = self.LFS_xcep.block_8_features(fea_LFS)
        fea_FAD, fea_LFS = self.Mixblock1(fea_FAD, fea_LFS)

        # Mixblock2
        fea_FAD = self.FAD_xcep.block_11_features(fea_FAD)
        fea_LFS = self.LFS_xcep.block_11_features(fea_LFS)
        fea_FAD, fea_LFS = self.Mixblock2(fea_FAD, fea_LFS)

        # Segment Decoder
        seg = self.Decoder(torch.cat((fea_FAD, fea_LFS), dim=1))

        # rest blocks
        fea_FAD = self.FAD_xcep.rest_blocks(fea_FAD)
        fea_LFS = self.LFS_xcep.rest_blocks(fea_LFS)
        
        # Cls
        fea_FAD = self._norm_fea(fea_FAD)
        fea_LFS = self._norm_fea(fea_LFS)
        y = torch.cat((fea_FAD, fea_LFS), dim=1)
        
        out = self.dp(y)
        out = self.fc(out)
        return seg,out



if __name__=="__main__":
    model = F3Net_Seg()
    print(model.Decoder.parameters())
    tmp = torch.randn(6,3,224,224)
    seg, out = model(tmp)
    print(seg.shape,out.shape)
    total_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(total_num)