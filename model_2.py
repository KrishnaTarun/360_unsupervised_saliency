import torch
from torchvision.models import vgg16
from torch import nn
from torch.nn import Dropout
#from torch.nn.functional import interpolate #Upsampling is supposedly deprecated, replace with interpolate, eventually, maybe
from torch.nn.modules.upsampling import Upsample
from torch.nn.functional import interpolate
from torch.autograd import Variable
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.activation import Sigmoid, ReLU
import torchvision





class Upsample(nn.Module):
    # Upsampling has been deprecated for some reason, this workaround allows us to still use the function within sequential.https://discuss.pytorch.org/t/using-nn-function-interpolate-inside-nn-sequential/23588
    def __init__(self, scale_factor, mode):
        super(Upsample, self).__init__()
        self.interp = interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class Multiply(nn.Module):
    def __init__(self):
        super(Multiply, self).__init__()
    def forward(self, tensors):
        prior = np.load('./prior.npy')
        prior = prior.astype(np.float32) 
        prior = torch.cuda.FloatTensor(prior)
        prior = prior.unsqueeze(0)
        return prior*tensors
class SalGAN_2(nn.Module):
    def  __init__(self):
        super(SalGAN_2,self).__init__()

        original_vgg16 = vgg16()

        self.encoder_salgan = torch.nn.Sequential(*list(original_vgg16.features)[:30])

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

        self.decoder_salgan = torch.nn.Sequential(*decoder_list)

    def forward(self, input_):
        latent = self.encoder_salgan(input_)  
        return self.decoder_salgan(latent)



class SalGAN_3(nn.Module):
    def  __init__(self):
        super(SalGAN_3,self).__init__()

        original_vgg16 = vgg16()

        self.encoder_salgan = torch.nn.Sequential(*list(original_vgg16.features)[:30])
        
        decoder_list=[
        Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        ReLU(),
        #Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #ReLU(),
        #Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #ReLU(),
        Upsample(scale_factor=2, mode='bilinear'),

        Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        ReLU(),
        #Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #ReLU(),
        #Dropout(p=0.2),
        #Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #ReLU(),
        Upsample(scale_factor=2, mode='bilinear'),

        Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        ReLU(),
        #Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #ReLU(),
        #Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #ReLU(),
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

        self.decoder_salgan = torch.nn.Sequential(*decoder_list)
        

    def forward(self, input_):
        latent = self.encoder_salgan(input_)  
        return self.decoder_salgan(latent)
        #return latent





class SalGAN_4(nn.Module):
    def  __init__(self):
        super(SalGAN_4,self).__init__()
        
        
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
        
        
        decoder_list=[
        Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        ReLU(),
        #Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #ReLU(),
        #Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #ReLU(),
        Upsample(scale_factor=2, mode='bilinear'),

        Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        ReLU(),
        #Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #ReLU(),
        #Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #ReLU(),
        Upsample(scale_factor=2, mode='bilinear'),

        Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        ReLU(),
        #Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #ReLU(),
        #Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #ReLU(),
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

        self.decoder_salgan = torch.nn.Sequential(*decoder_list)
        
        
        

    def forward(self, input_):
        latent = self.encoder_salgan(input_)
        return self.decoder_salgan(latent)
        #return latent

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
        total = torch.tensor(0)
        for i in range(0, inp.size()[0]):
            loss = _pointwise_loss(lambda a, b: self.KLD(a, b), inp[i], trg[i])
            total = loss + total.item()
        return torch.mean(total)

import os
import cv2
a = os.listdir('./')
for i in a:
     img = cv2.imread('./'+i)
     img = cv2.applyColorMap(img, 11)
     cv2.imwrite('./'+i,img)


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
        fix = fix > 0.1
        # Normalize saliency map to have zero mean and unit std
        sal_map = self.stand_normlize(sal_map)
        return sal_map[fix].mean()


class CC_Loss(nn.Module):
    def __init__(self):
        super(CC_Loss, self).__init__()
    def normliz(self ,x):
        return (x - x.mean())/x.std()

    def forward(self,saliency_map,gtruth):
        total = torch.tensor(0)
        for i in range(0, gtruth.size()[0]):
            saliency = self.normliz(saliency_map[i])
            gt = self.normliz(gtruth[i])
            total = total+torch.sum(saliency * gt) / (torch.sqrt(torch.sum(saliency ** 2)) * torch.sqrt(torch.sum(gt ** 2)))
        return torch.mean(total)

# in this implementation we consider standatrd normalization 


class JSD_Loss(nn.Module):
    def __init__(self):
        super(JSD_Loss, self).__init__()

    def forward(self,p, q):
        
        total = torch.tensor(0)
        eps = sys.float_info.epsilon
        for i in range(0, p.size()[0]):
            gt = p[i].squeeze(0)
            pred = q[i].squeeze(0)
            
            t1 = torch.mean(gt * torch.log(eps+torch.abs(torch.div((2*gt),(pred+gt)))))
            t2 =  torch.mean(pred * torch.log(eps+torch.abs(torch.div((2*pred), (pred+gt)))))
            t3 = torch.mean((gt-1) * torch.log(torch.abs(torch.div(((gt-1)), (pred+gt-2)))+eps))
            t4 = torch.mean((pred-1) * torch.log(torch.abs(torch.div(((pred-1)),(pred+gt-2)))+eps))
            JSD = (t1+t2-t3-t4)/2
            total = JSD + total.item()
        return torch.mean(total)

class LOSS(nn.Module):
    def __init__(self):
        super(LOSS, self).__init__()
        self.KLDLoss = KLDLoss()
        self.NSSLoss = NSSLoss()
        self.CC_Loss = CC_Loss()
        self.JSD_Loss = JSD_Loss()

    def forward(self,saliency_map , gtruth, fx):
        
        loss = self.KLDLoss(saliency_map,gtruth) 
        
        return loss 