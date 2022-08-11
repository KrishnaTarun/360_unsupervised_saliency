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
from torchvision import transforms, utils
from skimage.transform import resize



class VGG(nn.Module):

    def  __init__(self):
          super(VGG,self).__init__()
      
          original_vgg16 = vgg16()

          self.encoder_salgan = torch.nn.Sequential(*list(original_vgg16.features)[:30])
    

    def forward(self,input):
        # nn.DataParallel(self.encoder)
        bottel_neck = self.encoder_salgan(input)     

        
        
        return bottel_neck