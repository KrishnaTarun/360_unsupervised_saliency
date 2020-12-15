import torch
from torch import nn, sigmoid
from torchvision.models import vgg16
from torch.nn import MaxPool2d,BatchNorm2d
from torch.nn.modules.upsampling import Upsample
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.activation import Sigmoid, ReLU
from torch.nn import Conv3d, MaxPool3d, BatchNorm3d, ConvTranspose3d
import sys
import torchvision
from torch.nn.functional import interpolate
from torchsummary import summary
import torch.nn.functional as F



class ResNetEncoder(nn.Module):
  def __init__(self,):
    super(ResNetEncoder, self).__init__()

    model = torchvision.models.resnet50(pretrained=False, progress=False)

    Enc = torch.nn.Sequential(*list(model.children())[:-1])
    self.encoder1 = torch.nn.Sequential(*list(Enc.children())[:7])
    self.encoder2 = torch.nn.Sequential(*list(Enc.children())[7:])
    del model
    del Enc
    self.fc1 = nn.Linear(in_features=2048, out_features=512)
    self.relu =nn.ReLU()
    self.bnorm = nn.BatchNorm1d(512)

  def forward(self, in_, flag='map'):
      
      
      #No Normalization in this module
      if flag =='fc_map':
        Zx = self.encoder1(in_)
        fx  = F.relu(self.bnorm(self.fc1(self.encoder2(Zx).view(in_.size()[0],-1).contiguous())))
        return fx, Zx

      if flag=='map':
        Zt = self.encoder1(in_)
        return Zt
      
class SelfAttLocDim(nn.Module):
  def __init__(self, device, feat_dim=512):

        super(SelfAttLocDim, self).__init__()
        
        self.encoder = ResNetEncoder()
        self.feat_dim = feat_dim
        self.localdim = LocalDistractor(in_channel=512, in_fc=512, out_fc=512)
        self.attention = SelfAttention(in_channel=1024, K=8)        
        if device != 'cpu':
           self.encoder = torch.nn.DataParallel(self.encoder)#can do better
          # self.localdim = nn.DataParallel(self.localdim)

  def forward(self, x, z):
    
    x, zx = self.encoder(x, flag = 'fc_map') 
    z = self.encoder(z, flag = 'map')

    z, gamma = self.attention(zx, z)
    x_dim, z_dim = self.localdim(x, z)

    return x_dim, z_dim, gamma 

#TODO (uncomment) without self attention--> look into it inorder to run without self attention
# class LocalEncoder(nn.Module):
#     #pass your resnet model in here
#   def __init__(self, device, feat_dim=512):
#         super(LocalEncoder, self).__init__()
        
#         self.encoder = ResNetEncoder(3, feat_dim)
#         self.feat_dim = feat_dim
#         self.localdim = LocalDistractor(in_channel=512, in_fc=512, out_fc=512)
        
#         if device != 'cpu':
#           self.encoder = nn.DataParallel(self.encoder)
#           # self.localdim = nn.DataParallel(self.localdim)

#   def forward(self, x, z,layer=32):
    
#     x = self.encoder(x, layer=4) #normalized 
#     z = self.encoder(z, layer=1) #maps

#     z_dim = self.localdim(x, z)

#     return x, z_dim 

class LocalDistractor(nn.Module):
  
  #local encoder-dot-architecture
  def __init__(self, in_channel=512, in_fc=512, out_fc=512):
    super(LocalDistractor, self).__init__()

    self.fc_net = nn.Linear(in_features=512, out_features=512) #use this to encode f(x), previously this was in salgan encoder follwed by normalization
    self.f0_conv = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(1, 1))
    self.f1_conv = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(1, 1))
    self.b_norm = nn.BatchNorm2d(num_features=in_channel) 
    

    
  def forward(self, x, z):
    
    b, C, h, w = z.size()
    
    out0 = F.relu(self.fc_net(x))
    out0 = F.normalize(out0, p=2, dim=1) #Normalization of g(f(x))

    out = F.relu(self.f0_conv(z))
    out = F.relu(self.b_norm(self.f1_conv(out)))
    out = F.normalize(out, p=2, dim=1)#TODO check this normalization
    # out = F.normalize(torch.div(out.view(b, C, h*w).sum(-1),h*w),p=2, dim=1)

    # x = F.relu(self.fc_net(x))


    return out0, out
        
    
class SelfAttention(nn.Module):

  def __init__(self, in_channel = 1024, K=8):
    
    super(SelfAttention, self).__init__()

    self.in_channel = in_channel
    self.K = 8
    self.qry_ =  nn.Conv2d(in_channels=in_channel, out_channels= int(in_channel/K), kernel_size = (1,1))
    self.key_ =  nn.Conv2d(in_channels=in_channel, out_channels=int(in_channel/K), kernel_size = (1,1))
    self.val_ =  nn.Conv2d(in_channels=in_channel, out_channels= int(in_channel/2), kernel_size = (1,1))
    self.out  =  nn.Conv2d(in_channels=int(in_channel/2), out_channels=in_channel, kernel_size = (1,1))
    self.gamma = nn.Parameter(torch.zeros(1)*1.0)

    self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

  def forward(self, zx, z):


    b, c, h, w = zx.size()
    qr  = self.qry_(zx).view(b, int(self.in_channel/self.K), -1)
    key = self.maxpool(self.key_(z))
    key = key.view(b, int(self.in_channel/self.K), -1)
    
    attn = torch.bmm(qr.permute(0,2,1), key)
    attn = F.softmax(attn, dim=-1)

    val = self.maxpool(self.val_(z)).view(b, int(self.in_channel/2), -1)

    att_val = torch.bmm(attn, val.permute(0,2,1)).permute(0, 2,1)
    att_val = torch.reshape(att_val, (b, int(self.in_channel/2), h, w))
    out = self.out(att_val)#input to dim block
    
    #uncomment for the moment
    out = self.gamma * out + z 

    return out, self.gamma


class Normalize(nn.Module):

    def __init__(self, p=2):
        super(Normalize, self).__init__()
        self.p = p

    def forward(self, x):
        return F.normalize(x, p=self.p, dim=1)
# ================== Ends ========

if __name__ =="__main__":


  x = torch.rand((4, 3, 160, 320))
  y = torch.rand((4, 3, 160, 320))
  model = SelfAttLocDim(device='cuda')
  model = model.cuda()
  x = x.cuda()
  y = y.cuda()
  g = model(x, y)
  print(g.size())



            
            
            
