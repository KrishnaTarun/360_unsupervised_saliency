from os import O_NDELAY, pread
import torch
from torchvision.models import vgg16    
from torch import nn
from torch.autograd import Variable
from torch.nn import MaxPool2d
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.activation import  ReLU
from torch.nn import BatchNorm2d
from skimage.transform import resize
import numpy as np
import cv2
import torchvision



class Downsample(nn.Module):
    # specify the kernel_size for downsampling 
    def __init__(self, kernel_size):
        super(Downsample, self).__init__()
        self.pool = MaxPool2d(kernel_size)
        
        
    def forward(self, x):
        x = self.pool(x)
        return x


def make_conv_layers(cfg):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M_2':
            layers += [Downsample(kernel_size= 2)]
        elif v == 'M_4':
            layers += [Downsample(kernel_size= 4)]
        else:
            conv = Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv, ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)





from torch import nn, sigmoid
from torch.nn.modules.upsampling import Upsample
from torch.nn.functional import interpolate, dropout2d
from torch.autograd import Variable
from torch.nn import MaxPool2d
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.activation import Sigmoid, ReLU

#from Encoders import  based_AM

# create pooling layer
class Downsample(nn.Module):
    # specify the kernel_size for downsampling 
    def __init__(self, kernel_size):
        super(Downsample, self).__init__()
        self.pool = MaxPool2d(kernel_size)
        
        
    def forward(self, x):
        x = self.pool(x)
        return x

# create unpooling layer 
class Upsample(nn.Module):
    # specify the scale_factor for upsampling 
    def __init__(self, scale_factor, mode):
        super(Upsample, self).__init__()
        self.interp = interpolate
        self.scale_factor = scale_factor
        self.mode = 'bilinear'

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners = True)
        return x

# create Add layer , support backprop
class Multiply(nn.Module):
    def __init__(self):
        super(Multiply, self).__init__()
    def forward(self, tensors):
        prior = np.load('./prior.npy')
        prior = prior.astype(np.float32) 
        prior = torch.cuda.FloatTensor(prior)
        prior = prior.unsqueeze(0)
        #result = torch.ones(tensors[0].shape).cuda()
        #for t in tensors:
        #    result *= t
        return prior*tensors

# create Multiply layer , supprot backprop
class Add(nn.Module):
    def __init__(self):
        super(Add, self).__init__()

    def forward(self, tensors):
        result = torch.zeros(tensors[0].shape).cuda()
        for t in tensors:
            result += t
        return result


# reshape vectors layer
class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class SalGAN(nn.Module):

    def  __init__(self):
          super(SalGAN,self).__init__()
          #original_vgg16 = vgg16()
          model = torchvision.models.resnet50(pretrained=False, progress=False)

          encoder = torch.nn.Sequential(*list(model.children())[:-1])
          #encoder = torch.nn.Sequential(*list(original_vgg16.features)[:30])
    
          self.encoder_salgan = torch.nn.Sequential(*(list(encoder.children()))[:7])
        #   self.encoder_salgan = nn.DataParallel(self.encoder_salgan)

          decoder_salgan =[
          Conv2d(1024, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
          ReLU(),            
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
          Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
          Sigmoid(),
          ]

          self.decoder_salgan = torch.nn.Sequential(*decoder_salgan)
        #   self.decoder_salgan = nn.DataParallel(self.decoder_salgan)

    def forward(self,input):
        # nn.DataParallel(self.encoder)
        bottel_neck = self.encoder_salgan(input)
        
        output = self.decoder_salgan(bottel_neck)
        
        return output
        

        






if __name__=='__main__':

    model = SalGAN()
    # print(model_dict.keys())
    weight = torch.load('models/encode_Local_init_rand_memory_nce_16000_lr_0.01_decay_0.0001_bsz_20_optim_SGD/ckpt_epoch_138.pth')
    # model_dict =  model.encoder_salgan.state_dict()
    model_dict =  model.state_dict()
    # print(len(model_dict.keys()))
    # print(model_dict.keys())
    pretrained_dict = {"encoder_salgan."+'.'.join(k.split('encoder1')[-1].split('.')[1:]):
                         v for k, v in weight['model'].items() if 'encoder1' in k.split('.')}
    model_dict.update(pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    # print(len(model_dict.keys()))
    # # print(pretrained_dict.keys())
    # print()
    # print(model_dict.keys())
    
    

