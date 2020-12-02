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
    
class SelfAttLocDim(nn.Module):
  def __init__(self, device, feat_dim=512):
        super(SelfAttLocDim, self).__init__()
        
        self.encoder = SalGan(3, feat_dim)
        self.feat_dim = feat_dim
        self.localdim = LocalDistractor(in_channel=512, in_fc=512, out_fc=512)
        self.attention = SelfAttention(in_channel=512, K=8)        
        if device != 'cpu':
          self.encoder = nn.DataParallel(self.encoder)
          # self.localdim = nn.DataParallel(self.localdim)

  def forward(self, x, z,layer=32):
    
    x, zx = self.encoder(x, layer=4, flag=True) #normalized(x) 
    z = self.encoder(z, layer=1) #maps
    z = self.attention(zx, z)
    z_dim = self.localdim(z)

    return x, z_dim 

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
    self.b_norm = nn.BatchNorm2d(num_features=in_channel) 
    

    
  def forward(self, z):
    
    b, C, h, w = z.size()
    
    out = F.relu(self.f0_conv(z))
    out = F.relu(self.b_norm(self.f1_conv(out)))
    out = F.normalize(out, p=2, dim=1)
    # out = F.normalize(torch.div(out.view(b, C, h*w).sum(-1),h*w),p=2, dim=1)

    # x = F.relu(self.fc_net(x))


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

    self.fc7 = nn.Sequential(
      # 
      nn.Linear(feat_dim, feat_dim),
      #nn.BatchNorm1d(feat_dim),
      nn.ReLU(inplace=True),
      )  

    self.l2norm = Normalize(2)

  def forward(self, x, layer=1, flag=False):

    out = self.salgan(x)
    if flag:
      out_ = out
    if layer==1:
      return out
    out = self.conv_block_5_2(out)
    
    
    if layer==2:
      return out
    out = out.view(x.shape[0], -1)
    out = self.fc6(out)
    
    if layer==3:
      return F.normalize(out, p=2, dim=1)
    
    out = self.fc7(out)
    out = F.normalize(out, p=2, dim=1)
    return  out, out_        
    
class SelfAttention(nn.Module):

  def __init__(self, in_channel = 512, K=8):
    
    super(SelfAttention, self).__init__()

    self.in_channel = in_channel
    self.K = 8
    self.qry_ =  nn.Conv2d(in_channels=in_channel, out_channels= int(in_channel/K), kernel_size = (1,1))
    self.key_ =  nn.Conv2d(in_channels=in_channel, out_channels=int(in_channel/K), kernel_size = (1,1))
    self.val_ =  nn.Conv2d(in_channels=in_channel, out_channels= int(in_channel/2), kernel_size = (1,1))
    self.out  =  nn.Conv2d(in_channels=int(in_channel/2), out_channels=in_channel, kernel_size = (1,1))
    self.gamma = nn.Parameter(torch.zeros(1))

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
    
    out = self.gamma * out + z

    return out 


class Normalize(nn.Module):

    def __init__(self, p=2):
        super(Normalize, self).__init__()
        self.p = p

    def forward(self, x):
        return F.normalize(x, p=self.p, dim=1)
# ================== Ends ========

if __name__ =="__main__":


  x = torch.rand(3, 512, 10, 20)
  y = torch.rand(3, 512, 10, 20)
  model = SelfAttention()
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
            
            
            
