import torch
from torch import nn, sigmoid
from torchvision.models import vgg16
from torch.nn import MaxPool2d,BatchNorm2d
from torch.nn.modules.upsampling import Upsample
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.activation import Sigmoid, ReLU
from torch.nn import Conv3d, MaxPool3d, BatchNorm3d, ConvTranspose3d
import sys
from torch.nn.functional import interpolate
from torchsummary import summary
import torch.nn.functional as F


class LocalEncoder(nn.Module):
  
  def __init__(self, device, feat_dim=512):
        super(LocalEncoder, self).__init__()
        
        self.encoder = SalGan(3, feat_dim)
        self.feat_dim = feat_dim
        self.localdim = LocalDistractor(in_channel=512, in_fc=512, out_fc=512)
        
        if device != 'cpu':
          self.encoder = nn.DataParallel(self.encoder)
          # self.localdim = nn.DataParallel(self.localdim)

  def forward(self, x, z,layer=32):
    
    x = self.encoder(x, layer=4) #normalized 
    z = self.encoder(z, layer=1) #maps

    z_dim = self.localdim(z)

    return x, z_dim 

class LocalDistractor(nn.Module):
  
  #local encoder-dot-architecture
  def __init__(self, in_channel=512, in_fc=1024, out_fc=512):
    super(LocalDistractor, self).__init__()

    # self.fc_net = nn.Linear(in_features=in_fc, out_features=out_fc)
    self.f0_conv = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=(1, 1))
    self.f1_conv = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=(1, 1))
    
    
  def forward(self,z):
    
    b, C, h, w = z.size()
    
    out = F.relu(self.f0_conv(z))
    out = F.relu(self.f1_conv(out))
    out = F.normalize(out.view(b, C, h*w).sum(-1),p=2, dim=1)

    return out
    
    


class VanillaEncoder(nn.Module):

    def __init__(self, device, feat_dim=1024):
        super(VanillaEncoder, self).__init__()
        
        self.encoder = SalGan(3, feat_dim)
        self.feat_dim = feat_dim

        if device != 'cpu':
          self.encoder = nn.DataParallel(self.encoder)

    def forward(self, x, z):

        return self.encoder(x), self.encoder(z)

class SalGan(nn.Module):

  def __init__(self, in_channel, feat_dim=512):
    super(SalGan,self).__init__()


    original_vgg16 = vgg16()
    
    # select only convolutional layers
    encoder = torch.nn.Sequential(*list(original_vgg16.features)[:30])
    
    # assamble the full architecture encoder-decoder
    self.salgan = torch.nn.Sequential(*(list(encoder.children())))
    
    #add further blocks
    self.conv_block_5_2 = nn.Sequential(
        Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        nn.BatchNorm2d(512),
        ReLU(inplace=True),
        MaxPool2d(kernel_size=2, stride=1, padding=0, dilation=1, ceil_mode=False),
    )

    self.fc6 = nn.Sequential(
      # 
      nn.Linear(512 * 4 * 9, feat_dim),
      nn.BatchNorm1d(feat_dim),
      nn.ReLU(inplace=True),
      )

    self.l2norm = Normalize(2)

  def forward(self, x, layer=1):

    out = self.salgan(x)
    if layer==1:
      return out
    out = self.conv_block_5_2(out)
    
    if layer==2:
      return out
    out = out.view(x.shape[0], -1)
    out = self.fc6(out)
    
    if layer==3:
      return out
    out = F.normalize(out, p=2, dim=1)
    return  out        



    



# class SelfAttention(nn.Module):
#   def __init__(self, in_channel = 512, K=8):
#     super(SelfAttention, self).__init__()

#     qry_ =  nn.Conv2d(in_channels=in_channel, out_channels=in_channel/K, kernel_size = (1,1))
#     key_ =  nn.Conv2d(in_channels=in_channel, out_channels=in_channel/K, kernel_size = (1,1))
#     val_ =  nn.Conv2d(in_channels=in_channel, out_channels=in_channel/K, kernel_size = (1,1))




class Normalize(nn.Module):

    def __init__(self, p=2):
        super(Normalize, self).__init__()
        self.p = p

    def forward(self, x):
        return F.normalize(x, p=self.p, dim=1)
# ================== Ends ========

if __name__ =="__main__":


  x = torch.rand(3, 7, 1024)
  y = torch.rand(3, 512, 4, 9)
  model = LocalDim()
  g = model(x,y)
  print(g.size())


#     # model = Encoder('cuda')
#     # model = Encoder('cuda') 
    
#     """
#       Keep them
#     """
#     # for j, layer in enumerate(model.encoder.module.children()):
#     #   print(j, layer)
#     #   try:
#     #     if (j==0):
#     #       for i, k in enumerate(layer.modules()):
            
#     #           print(i, k)
#     #           if isinstance(k, nn.Conv2d):
#     #               k.weight.requires_grad=False
#     #               k.bias.requires_grad=False
#     #   except IndexError as e:
#     #     pass  

#     weight = torch.load("./salgan_3dconv_module.pt")  
#     # print(model.encoder.module.encoder_salgan.state_dict())  
#     # model.encoder.module.encoder_salgan.load_state_dict(weight, strict=False)
#     # print(model.encoder.module.encoder_salgan.state_dict())  
#     """
#     """
    



#     # with torch.no_grad():
#     cout=0
#     for k, m in weight.items():
#       if(k.split('_')[0]=='encoder'):
#         if(k.split('.')[-1]=='weight'):
#           print(cout , k, m.size())
#           cout+=1
#       # print(i, m)
            
            
            
