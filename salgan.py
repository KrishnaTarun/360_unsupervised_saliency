import torch
from torch import nn, sigmoid
from torch.nn import MaxPool2d
from torch.nn.modules.upsampling import Upsample
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.activation import Sigmoid, ReLU
from torchvision.models import vgg16
from torch import cat

class SalGAN_Generator(nn.Module):

    def  __init__(self):
          super(SalGAN_Generator,self).__init__()
          
          encoder_salgan =[
                            
          Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
          ReLU(),
          Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
          ReLU(),
          MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
          Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
          ReLU(),
          Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
          ReLU(),
          MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
          Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
          ReLU(),
          Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
          ReLU(),
          Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
          ReLU(),
          MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
          Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
          ReLU(),
          Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
          ReLU(),
          Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
          ReLU(),
          MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
          Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
          ReLU(),
          Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
          ReLU(),
          Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
          ReLU()]

          decoder_list=[
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Upsample(scale_factor=2, mode='bilinear'),

            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Upsample(scale_factor=2, mode='bilinear'),

            Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Upsample(scale_factor=2, mode='bilinear'),

            Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Upsample(scale_factor=2, mode='bilinear'),

            Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1), padding=0),
            Sigmoid(),
            ]

          #self.encoder_salgan = torch.nn.Sequential(*vgg16(pretrained=True).features[:30])
          self.encoder_salgan = torch.nn.Sequential(*encoder_salgan)
          self.decoder_salgan = torch.nn.Sequential(*decoder_list)

    def forward(self,input):

        bottel_neck = self.encoder_salgan(input)

        return self.decoder_salgan(bottel_neck)