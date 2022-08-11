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

import os
import cv2
a = os.listdir('./')
for i in a:
     img = cv2.imread('./'+i)
     img = cv2.applyColorMap(img, 11)
     cv2.imwrite('./'+i,img)

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
          """
          original_vgg16 = vgg16()

          self.encoder_salgan = torch.nn.Sequential(*list(original_vgg16.features)[:30])
          """
          model = torchvision.models.resnet50(pretrained=False, progress=False)

          encoder = torch.nn.Sequential(*list(model.children())[:-1])
          #encoder = torch.nn.Sequential(*list(original_vgg16.features)[:30])
          encoder1 = torch.nn.Sequential(*list(encoder.children())[:7])

          bot_1 = nn.Sequential(
              nn.Conv2d(1024,512, kernel_size=(1,1), stride=(1,1)),
              nn.BatchNorm2d(512),
              nn.Conv2d(512,512, kernel_size=(1,1),stride=(1,1)),
              nn.BatchNorm2d(512),
              nn.ReLU())


          self.encoder_salgan = torch.nn.Sequential(*(list(encoder1.children())+list(bot_1.children())))
          

    
        #  self.encoder_salgan = torch.nn.Sequential(*(list(encoder.children()))[:7])
        #   self.encoder_salgan = nn.DataParallel(self.encoder_salgan)
          


          decoder_salgan=[
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
            Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1), padding=0),
            Sigmoid(),
            ]

          self.decoder_salgan = torch.nn.Sequential(*decoder_salgan)
        #   self.decoder_salgan = nn.DataParallel(self.decoder_salgan)

    

    def forward(self,input):
        # nn.DataParallel(self.encoder)
        bottel_neck = self.encoder_salgan(input)     

        """    
        for i in range(0,100):
            saliency_map = bottel_neck.squeeze()[i]

            saliency_map = (saliency_map.cpu()).detach().numpy()

            saliency_map = resize(saliency_map, (160, 320))

            saliency_map = torch.FloatTensor(saliency_map)

            post_process_saliency_map = (saliency_map - torch.min(saliency_map)) / (
                                torch.max(saliency_map) - torch.min(saliency_map))
            utils.save_image(post_process_saliency_map,"./map/smap{}.png".format(i)) 

        """
        
        
        return bottel_neck
        

        





import sys

from torch.nn.functional import interpolate 

def _pointwise_loss(lambd, input, target, size_average=True, reduce=True):
    d = lambd(input, target)
    if not reduce:
        return d
    return torch.mean(d) if size_average else torch.sum(d)

class KLDLoss(nn.Module):
    def __init__(self):
        super(KLDLoss, self).__init__()

    def KLD(self, inp, trg):
        inp = inp/torch.sum(inp)
        trg = trg/torch.sum(trg)
        eps = sys.float_info.epsilon

        return torch.sum(trg*torch.log(eps+torch.div(trg,(inp+eps))))

    def forward(self, inp, trg):
        return _pointwise_loss(lambda a, b: self.KLD(a, b), inp, trg)



# inn this implementation we consider standatrd normalization 

class NSSLoss(nn.Module):
    def __init__(self):
        super(NSSLoss, self).__init__()

    #normalize saliency map
    def stand_normlize(self, x) :
       # res = (x - np.mean(x)) / np.std(x)
       # x should be float tensor 
       return (x - x.mean())/x.std()

    def forward(self, sal_map, fix):
        if sal_map.size() != fix.size():
           sal_map = interpolate(sal_map, size= (fix.size()[1],fix.size()[0]))
           print(sal_map.size())
           print(fix.size())
        # bool tensor 
        fix = fix > 0.5
        # Normalize saliency map to have zero mean and unit std
        sal_map = self.stand_normlize(sal_map)
        return sal_map[fix].mean()

class CC_Loss(nn.Module):
    def __init__(self):
        super(CC_Loss, self).__init__()
    def normliz(self ,x) :
        return (x - x.mean())
    def forward(self,saliency_map,gtruth):
         saliency_map = self.normliz(saliency_map)
         gtruth       = self.normliz(gtruth)
         return torch.sum(saliency_map * gtruth) / (torch.sqrt(torch.sum(saliency_map ** 2)) * torch.sqrt(torch.sum(gtruth ** 2)))

# in this implementation we consider standatrd normalization 

class LOSS(nn.Module):
    def __init__(self):
        super(LOSS, self).__init__()
        self.KLDLoss = KLDLoss()
        self.NSSLoss = NSSLoss()
        self.CC_Loss = CC_Loss()

    def forward(self,saliency_map , gtruth):
        
        loss = self.KLDLoss(saliency_map,gtruth)
        
        return loss

        
if __name__=='__main__':

    model = SalGAN()
    # weight = torch.load("../salgan_3dconv_module.pt")  
    # model.load_state_dict(weight, strict=False)
    # weight = torch.load('../models/memory_nce_16000_lr_0.01_decay_0.0001_bsz_22_optim_SGD/ckpt_epoch_130.pth')
    # print(weight['model'].keys())
    # print(model.state_dict().keys())
    # for k, v in weight[model].items():
    #     print(k)
    
    # for j, layer in enumerate(model.children()):
    #     print(j ,layer)
    #     try:
    #     #     # print(layer, j, model_keys[j][0].bias.shape)
            
    #         for id_, k in enumerate(model.encoder.modules()):
    #         # print(id_)
    #             if isinstance(k, nn.Conv2d):
    #                 print('hell')
    #                 # c+=1
        
    #     except IndexError as e:
    #         pass
    def init_weights(m):
        if type(m) == nn.Conv2d:
            print('yes')
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)
    
    model.decoder_salgan.apply(init_weights)

    # for params in model.decoder_salgan.children():

    #     if isinstance(params, nn.Conv2d):
    #         print('yes')
        # print(params)
        # if(params.requires_grad==True):
