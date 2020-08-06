import torch
from torch import nn, sigmoid
from torch.nn import MaxPool2d,BatchNorm2d
from torch.nn.modules.upsampling import Upsample
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.activation import Sigmoid, ReLU
from torch.nn import Conv3d, MaxPool3d, BatchNorm3d, ConvTranspose3d
import sys
from torch.nn.functional import interpolate
from torchsummary import summary


class Encoder(nn.Module):

    def __init__(self, device, feat_dim=1024):
        super(Encoder, self).__init__()
        self.encoder = SalEncoder(3, feat_dim)
        if device != 'cpu':
            self.encoder = nn.DataParallel(self.encoder)

    def forward(self, x, t1, t2, layer=7):
        z = torch.cat((x.unsqueeze(1), t1.unsqueeze(1), t2.unsqueeze(1)), dim=1)
        z = z.contiguous().view(-1, 3, 160, 320)

        z = self.encoder(z, layer)
        z = z.view(x.shape[0], -1, 1024)
        z1, z2, z3 = torch.split(z, 1, 1)

        return z1.squeeze(), z2.squeeze(), z3.squeeze()


class SalEncoder(nn.Module):
    """
    Encoder module for Contrastive Learning
    """

    def __init__(self, in_channel, feat_dim=1024):
        super(SalEncoder, self).__init__()
        
        #TODO: Batch_Normalization
        self.conv_block_1 = nn.Sequential(
            Conv2d(in_channel, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            ReLU(inplace=True),
            Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                    
            )

        self.conv_block_2 = nn.Sequential(
            
            Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            ReLU(inplace=True),
            Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            
            )

        self.conv_block_3 = nn.Sequential(
            
            Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            ReLU(inplace=True),
            Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            ReLU(inplace=True),
            Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            
            )

        self.conv_block_4 = nn.Sequential(
            Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            ReLU(inplace=True),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            ReLU(inplace=True),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            )

        self.conv_block_5_1 = nn.Sequential(
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            ReLU(inplace=True),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            ReLU(inplace=True),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            ReLU(inplace=True)
            )

        #extra layer (with change in it's stride)
        self.conv_block_5_2 = nn.Sequential(
            Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(512),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            )

        self.fc6 = nn.Sequential(
            # nn.Linear(512 * 5 * 10, 4096 ),
            nn.Linear(512 * 5 * 2, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
        )
        self.l2norm = Normalize(2)
        self.fc7 = nn.Sequential(
            # nn.Linear(4096, 4096),
            nn.Linear(1024, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),   
        )
        # self.fc8 = nn.Sequential(
        #     nn.Linear(4096, feat_dim),
        #     nn.BatchNorm1d(feat_dim),
        #     nn.ReLU(inplace=True),   
        # )
        self.l2norm = Normalize(2)

    
    def forward(self, input, layer=6):


      if layer <= 0:
        return input

      x = self.conv_block_1(input)
      if layer == 1:
        return x

      x = self.conv_block_2(x)
      if layer == 2:
        return x

      x = self.conv_block_3(x)
      if layer == 3:
        return x

      x = self.conv_block_4(x)
      if layer == 4:
        return x

      x = self.conv_block_5_1(x)
      if layer==5:
        return x 

      x = self.conv_block_5_2(x)
      if layer==5:
        return x
      x = x.view(x.shape[0], -1)
      x = self.fc6(x)
      x = self.l2norm(x)
      if layer==6:
        return x


      x  = self.fc7(x)
      # if layer==7:
      #   return x

      # x = self.fc8(x)
      x = self.l2norm(x)


      return x     

class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out
# ================== Ends ========

if __name__ =="__main__":


    model = SalEncoder(3,).to("cuda:0")
    summary(model, (3, 160, 320))
