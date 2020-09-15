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
import torch.nn.functional as F

class Encoder(nn.Module):

    def __init__(self, device, feat_dim=512):
        super(Encoder, self).__init__()
        
        self.encoder = SalGan(3, feat_dim)
        self.feat_dim = feat_dim

        if device != 'cpu':
            self.encoder = nn.DataParallel(self.encoder)

    def forward(self, x, z, layer=32):
        # ht, wt = x.shape[2], x.shape[3]
        # print(x.shape, ht, wt)
        # print(x.shape, t1.shape, t2.shape)
        # z = torch.cat((x.unsqueeze(1),
        #                t1.unsqueeze(1),
        #                t2.unsqueeze(1)),
        #                dim=1)

        # z = z.contiguous().view(-1, 3, ht, wt)
        # # print('here ==>', z.shape)/

        # z = self.encoder(z, layer)
        # # print('out==>', z.shape, layer)
        # z = z.view(x.shape[0], -1, self.feat_dim)
        # z1, z2, z3 = torch.split(z, 1, 1)
        # t1 = 
        # t2 =   
        return self.encoder(x), self.encoder(z)

class SalGan(nn.Module):

  def __init__(self, in_channel, feat_dim=512):
    super(SalGan,self).__init__()

    # encoder_salgan =[
                              
    #         Conv2d(in_channel, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    #         ReLU(),
    #         Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    #         ReLU(),
    #         MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
    #         Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    #         ReLU(),
    #         Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    #         ReLU(),
    #         MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
    #         Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    #         ReLU(),
    #         Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    #         ReLU(),
    #         Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    #         ReLU(),
    #         MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
    #         Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    #         ReLU(),
    #         Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    #         ReLU(),
    #         Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    #         ReLU(),
    #         MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
    #         Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    #         ReLU(),
    #         Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    #         ReLU(),
    #         Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    #         ReLU()]

    self.encoder_salgan = nn.Sequential(
              Conv2d(in_channel, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
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
              ReLU()
    )

    self.conv_block_5_2 = nn.Sequential(
        Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        # nn.BatchNorm2d(512),
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

  def forward(self, x, layer=32):

    # pass
    # print('===>', x.shape)
    # for cut in range(len(self.encoder_salgan)):

    #   if cut == layer:
    #       return self.encoder_salgan(x)

    #   if layer == 31:
    #         out = self.encoder_salgan(x)
    #         return self.conv_block_5_2(out)
      
    #   else :
    out = self.encoder_salgan(x)
    # print(out.shape)
    out = self.conv_block_5_2(out) 
    # print(out.shape)
    # out = out.view(1, -1)
    out = out.view(x.shape[0], -1)
    # print(out.shape)
    out =self.l2norm(self.fc6(out))
    # print(out.shape)
    return  out         



# class SalEncoder(nn.Module):
#     """
#     Encoder module for Contrastive Learning
#     """

#     def __init__(self, in_channel, feat_dim=1024):
#         super(SalEncoder, self).__init__()
        
#         #TODO: Batch_Normalization
#         self.conv_block_1 = nn.Sequential(
#             Conv2d(in_channel, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#             # nn.BatchNorm2d(64),
#             ReLU(inplace=True),
#             Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#             # nn.BatchNorm2d(64),
#             ReLU(inplace=True),
#             MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                    
#             )

#         self.conv_block_2 = nn.Sequential(
            
#             Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#             # nn.BatchNorm2d(128),
#             ReLU(inplace=True),
#             Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#             # nn.BatchNorm2d(128),
#             ReLU(inplace=True),
#             MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            
#             )

#         self.conv_block_3 = nn.Sequential(
            
#             Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#             # nn.BatchNorm2d(256),
#             ReLU(inplace=True),
#             Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#             # nn.BatchNorm2d(256),
#             ReLU(inplace=True),
#             Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#             # nn.BatchNorm2d(256),
#             ReLU(inplace=True),
#             MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            
#             )

#         self.conv_block_4 = nn.Sequential(
#             Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#             # nn.BatchNorm2d(512),
#             ReLU(inplace=True),
#             Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#             # nn.BatchNorm2d(512),
#             ReLU(inplace=True),
#             Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#             # nn.BatchNorm2d(512),
#             ReLU(inplace=True),
#             MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
#             )

#         self.conv_block_5_1 = nn.Sequential(
#             Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#             # nn.BatchNorm2d(512),
#             ReLU(inplace=True),
#             Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#             # nn.BatchNorm2d(512),
#             ReLU(inplace=True),
#             Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#             # nn.BatchNorm2d(512),
#             ReLU(inplace=True)
#             )

#         #extra layer (with change in it's stride)
#         self.conv_block_5_2 = nn.Sequential(
#             Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
#             nn.BatchNorm2d(512),
#             ReLU(inplace=True),
#             MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
#             )

#         self.fc6 = nn.Sequential(
#             # nn.Linear(512 * 5 * 10, 4096 ),
#             nn.Linear(512 * 5 * 2, feat_dim),
#             nn.BatchNorm1d(feat_dim),
#             nn.ReLU(inplace=True),
#         )
        
#         # self.fc7 = nn.Sequential(
#         #     # nn.Linear(4096, 4096),
#         #     nn.Linear(1024, feat_dim),
#         #     nn.BatchNorm1d(feat_dim),
#         #     nn.ReLU(inplace=True),   
#         # )
#         # self.fc8 = nn.Sequential(
#         #     nn.Linear(4096, feat_dim),
#         #     nn.BatchNorm1d(feat_dim),
#         #     nn.ReLU(inplace=True),   
#         # )
#         self.l2norm = Normalize(2)

    
#     def forward(self, input, layer=6):


#       if layer <= 0:
#         return input

#       x = self.conv_block_1(input)
#       if layer == 1:
#         return x

#       x = self.conv_block_2(x)
#       if layer == 2:
#         return x

#       x = self.conv_block_3(x)
#       if layer == 3:
#         return x

#       x = self.conv_block_4(x)
#       if layer == 4:
#         return x

#       x = self.conv_block_5_1(x)
#       if layer==5:
#         return x 

#       x = self.conv_block_5_2(x)
#       if layer==5:
#         return x
#       x = x.view(x.shape[0], -1)
      
#       x = self.fc6(x)
      
#       x = self.l2norm(x)
#       if layer==6:
#         return x


#       # x  = self.fc7(x)
#       # if layer==7:
#       #   return x

#       # x = self.fc8(x)
#       # x = self.l2norm(x)


#       return x     

class Normalize(nn.Module):

    def __init__(self, p=2):
        super(Normalize, self).__init__()
        self.p = p

    def forward(self, x):
        return F.normalize(x, p=self.p, dim=1)
# ================== Ends ========

if __name__ =="__main__":

    # model = Encoder('cuda')
    model = Encoder('cuda') 
    # summary(model, (3, 160, 320))
    # model_keys = [model.conv_block_1, model.conv_block_2, model.conv_block_3, model.conv_block_4, model.conv_block_5_1] 
    # net = torch.load("initial.pt")
    # copy_net_keys = []
    
    # # store keys
    # for k, v in net.items():
    #   # print("Layer {}".format(k))
      
    #   if 'encoder' in k.split('.'):
    #     copy_net_keys.append(k)
    # copy_net_keys = [tuple(copy_net_keys[i:i+2]) for i in range(0, len(copy_net_keys), 2)]
    # print(copy_net_keys)
    # c = 0
    # for j, layer in enumerate(model.children()):
    #   try:
    #     # print(layer, j, model_keys[j][0].bias.shape)
        
    #     for id_, k in enumerate(layer.modules()):
    #       # print(id_)
    #       if isinstance(k, nn.Conv2d):
    #         with torch.no_grad():
    #           model_keys[j][id_-1].weight.copy_(net[copy_net_keys[c][0]])
    #           model_keys[j][id_-1].bias.copy_(net[copy_net_keys[c][1]])
    #         print(c)
    #         c+=1
    
    #   except IndexError as e:
    #     pass
    for j, layer in enumerate(model.encoder.module.children()):
      print(j, layer)
      try:
        if (j==0):
          for i, k in enumerate(layer.modules()):
            
              print(i, k)
              if isinstance(k, nn.Conv2d):
                  k.weight.requires_grad=False
                  k.bias.requires_grad=False
      except IndexError as e:
        pass  

    # for params in model.parameters():
    #   # print(params)
    #   if(params.requires_grad==True):
    #     print('yes')



    # weight = torch.load("./salgan_3dconv_module.pt")  
    # print(model.encoder.module.encoder_salgan.state_dict())  
    # model.encoder.module.encoder_salgan.load_state_dict(weight, strict=False)
    # print(model.encoder.module.encoder_salgan.state_dict())  