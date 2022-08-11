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
          original_vgg16 = vgg16()
          encoder = torch.nn.Sequential(*list(original_vgg16.features)[:30])
    
        # # assamble the full architecture encoder-decoder
        # self.encoder = torch.nn.Sequential(*(list(encoder.children())))
        #   encoder_salgan =[
                            
        #   Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #   ReLU(),
        #   Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #   ReLU(),
        #   MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        #   Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #   ReLU(),
        #   Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #   ReLU(),
        #   MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        #   Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #   ReLU(),
        #   Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #   ReLU(),
        #   Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #   ReLU(),
        #   MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        #   Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #   ReLU(),
        #   Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #   ReLU(),
        #   Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #   ReLU(),
        #   MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        #   Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #   ReLU(),
        #   Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #   ReLU(),
        #   Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #   ReLU()]

          #self.encoder_salgan = torch.nn.Sequential(*vgg16(pretrained=True).features[:30])
          self.encoder_salgan = torch.nn.Sequential(*(list(encoder.children())))

          decoder_salgan =[
                           
          Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
          ReLU(),
          Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
          ReLU(),
          Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
          ReLU(),
          Upsample(scale_factor=2, mode='nearest'),

          Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
          ReLU(),
          Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
          ReLU(),
          Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
          ReLU(),
          Upsample(scale_factor=2, mode='nearest'),

          Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
          ReLU(),
          Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
          ReLU(),
          Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
          ReLU(),
          Upsample(scale_factor=2, mode='nearest'),

          Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
          ReLU(),
          Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
          ReLU(),
          Upsample(scale_factor=2, mode='nearest'),

          Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
          ReLU(),
          Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
          ReLU(),
          Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
          Sigmoid()
          ]

          self.decoder_salgan = torch.nn.Sequential(*decoder_salgan)

    def forward(self,input):

        bottel_neck = self.encoder_salgan(input)

        output      = self.decoder_salgan(bottel_neck)

        return output

model = SalGAN_Generator()
model.cuda()
from torchsummary import summary
summary(model, (3, 224, 224))



import cv2
from torchvision import transforms, utils
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision

X = cv2.imread('/home/tarun/Documents/Phd/360_cvpr/evaluate/dataset/training_set/image/14.jpg')
X_shape = X.shape
X = cv2.resize( X, (320, 160)).astype(np.float32)

X = torch.FloatTensor(X)
X = X.permute(2,0,1) 
torchvision.utils.save_image(torchvision.utils.make_grid(X, nrow=1), fp='X.jpg') 
X = X.cuda()

print(X.shape)

model = SalGAN_Generator()
# salgan weight
weight = torch.load("../salgan_3dconv_module.pt")
model.load_state_dict(weight,strict=False)
model.cuda()
with torch.no_grad():
  out = model(X.unsqueeze(0))
print(type(out.squeeze().cpu().numpy()))
out = out.squeeze().cpu().numpy()
cv2.imwrite( "image.png",255*out/out.max())
# Image.fromarray((255*out/out.max()).astype('uint8'))