import torch
import torch.nn as nn
import torch.nn.functional as F
# from alias_multinomial import AliasMethod
import math
from .alias_multinomial import AliasMethod
eps = 1e-7

# class LocalDim(nn.Module):
  
#   #local encoder-dot-architecture
#   def __init__(self, in_channel=512, in_fc=1024, out_fc=512):
#     super(LocalDim, self).__init__()

#     self.fc_net = nn.Linear(in_features=in_fc, out_features=out_fc)
#     self.f_conv = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=(1, 1))
#     self.bn_norm = nn.BatchNorm2d(num_features=in_channel)
    
#   def forward(self,x, mI, z):
#     """
#         mI size =  (b, K+1, 1024)
#         x = (b, 1024)
#         z = (b, c, h, w) feature maps for tramsformed images
#     """
#     b, C, h, w = z.size()
#     _, K, _  = mI.size()
    
#     #gVec from memory representation
#     gVec = F.relu(self.fc_net(mI))
#     #xVec from current representation
#     xVec = F.relu(self.fc_net(x)) 

#     lMap = F.relu(self.bn_norm(self.f_conv(z)))
#     # ------------Normalization------------------
#     gVec = F.normalize(gVec, dim=-1)
#     xVec = F.normalize(xVec, dim=-1)
#     lMap = F.normalize(lMap, dim=1)
#     #---------------------------------------------

#     #basically the logit for z representations through Local Dim Network
#     lz = torch.bmm(gVec, lMap.view(b, C, -1))
#     lz = lz.mean(-1)

#     return xVec, gVec, lz


class BaseMem(nn.Module):
    """Base Memory Class"""
    def __init__(self, K=65536, T=0.07, m=0.5):
        super(BaseMem, self).__init__()
        self.K = K
        self.T = T
        self.m = m

    def _update_memory(self, memory, x, y, Normalize=True):
        """
        Args:
          memory: memory buffer
          x: features
          y: index of updating position
        """
        with torch.no_grad():
            x = x.detach()
            w_pos = torch.index_select(memory, 0, y.view(-1))
            w_pos.mul_(self.m)
            w_pos.add_(torch.mul(x, 1 - self.m))
            updated_weight = w_pos
            if Normalize:
                updated_weight = F.normalize(w_pos, p=2, dim=1)
            memory.index_copy_(0, y, updated_weight)

    def _compute_logit(self, x, w):
        """
        Args:
          x: feat, shape [bsz, n_dim]
          w: softmax weight, shape [bsz, self.K + 1, n_dim]
        """
        x = x.unsqueeze(2)
        out = torch.bmm(w, x)
        out = torch.div(out, self.T)
        out = out.squeeze().contiguous()
        return out


class MemOps(BaseMem):

    def __init__(self, n_dim, n_data, K=65536, T=0.07, m=0.5, use_softmax=False):
        super(MemOps, self).__init__(K, T, m)
        
        # create sampler
        self.multinomial = AliasMethod(torch.ones(n_data))
        self.multinomial.cuda()
        
        # stdv = 1. / math.sqrt(n_dim / 3)

        # create memory bank
        self.register_buffer('memory', torch.randn(n_data, n_dim))
        self.register_buffer('params', torch.tensor([n_data, n_dim, -1]))

        self.memory = F.normalize(self.memory, p=2, dim=1)
        self.use_softmax = use_softmax
        
        # self.loss = NCESoftmaxLoss() 
    
    def forward(self, x, z, y):
        """
            x: untransformed image representation
            z: projected image representation
            y: index of x from Data-set

            L = (1-lambda)*L_nce(m, x) + (lambda)*L_nce(m, z)
        """
        bsz = x.shape[0]
        n_data = self.params[0].item()
        n_dim  = self.params[1].item()
        z0 = self.params[2].item()
        # z2 = self.params[3].item()
        
        #get representations from memory for x
        mI = torch.index_select(self.memory,0, y).detach()
        
        # print(y.shape, y.view(-1).shape)
        # print(self.memory.shape, mI.shape) 
        
        #samples indexes (Negative K samples)
        idx = self.multinomial.draw(bsz * (self.K)).view(bsz, -1)
        
        #get representations for negative samples
        w = torch.index_select(self.memory, 0 , idx.view(-1)).detach()
        w = w.view(bsz, self.K, n_dim)

        #combine mI and w
        mI = torch.cat((mI.unsqueeze(1),w), dim=1)

        #compute logit for x and z
        lx = self._compute_logit(x, mI)
        lz = self._compute_logit(z, mI)

        if not self.use_softmax:
            lx = torch.exp(lx)
            lz = torch.exp(lz)
            
            #refer instance discrimination for this
            if z0 < 0:
                self.params[2] = (lx.mean())*n_data
                z0 = self.params[2].clone().detach().item()
                print("normalization constant Z_0 is set to {:.1f}".format(z0))
            
            # if z2 < 0:
            #     self.params[3] = (lz.mean())*n_data
            #     z2 = self.params[3].clone().detach().item()
            #     print("normalization constant Z_2 is set to {:.1f}".format(z2))

            lx = torch.div(lx, z0).contiguous()
            lz = torch.div(lz, z0).contiguous()
        
        #update memory with current feature of x
        self._update_memory(self.memory, x, y) 

        return lx, lz
        
class MemOpsLocDim(BaseMem):

    def __init__(self, n_dim, n_data, K=65536, T=0.07, m=0.5, use_softmax=False):
        super(MemOpsLocDim, self).__init__(K, T, m)

        
        
        # create sampler
        self.multinomial = AliasMethod(torch.ones(n_data))
        self.multinomial.cuda()
        
        # stdv = 1. / math.sqrt(n_dim / 3)

        # create memory bank
        self.register_buffer('memory', torch.randn(n_data, n_dim))
        self.register_buffer('params', torch.tensor([n_data, n_dim, -1]))

        self.memory = F.normalize(self.memory, p=2, dim=1)
        self.use_softmax = use_softmax

    def forward(self, x, z, y):
        """
            x: untransformed image representation
            z: projected image representation (here this is a volume i.e feature maps)
            y: index of x from Data-set

            L = (1-lambda)*L_nce(m, x) + (lambda)*L_nce(m, z)
        """
        bsz = x.shape[0]
        n_data = self.params[0].item()
        n_dim  = self.params[1].item()
        z0 = self.params[2].item()
        # b, c, _, _ = z.size()
        # z2 = self.params[3].item()
        
        #get representations from memory for x
        mI = torch.index_select(self.memory,0, y).detach()
                        
        #samples indexes (Negative K samples)
        idx = self.multinomial.draw(bsz * (self.K)).view(bsz, -1)
        
        #get representations for negative samples
        w = torch.index_select(self.memory, 0 , idx.view(-1)).detach()#detach (no_gradients to memory)
        w = w.view(bsz, self.K, n_dim)

        
      
        #combine mI and w
        mI = torch.cat((mI.unsqueeze(1),w), dim=1)  
        # xVec, gVec, lz = self.localDim(x, mI, z)

        # #compute logit just for x
        #ssample indexes from z
        # i = torch.LongTensor(b).random_(0, w_*h_-1).cuda()
        # z = z.reshape((b, w_*h_, c))
        # #select index
        # z = z[torch.arange(b), i, :]
        
        # zz = F.normalize(z,  dim=-1)
        
        lx = self._compute_logit(x, mI)
        lz = self._compute_logit(z, mI)
        # lz = torch.div(lz, self.T)

        if not self.use_softmax:
            lx = torch.exp(lx)
            lz = torch.exp(lz)
            
            #refer instance discrimination paper for this
            if z0 < 0:
                self.params[2] = (lx.mean())*n_data
                z0 = self.params[2].clone().detach().item()
                print("normalization constant Z_0 is set to {:.1f}".format(z0))
            
            # if z2 < 0:
            #     self.params[3] = (lz.mean())*n_data
            #     z2 = self.params[3].clone().detach().item()
            #     print("normalization constant Z_2 is set to {:.1f}".format(z2))

            lx = torch.div(lx, z0).contiguous()
            lz = torch.div(lz, z0).contiguous()
        
        #update memory with current feature of x
        self._update_memory(self.memory, x, y, Normalize=True) 

        return lx, lz
    

class NCECriterion(nn.Module):
    """
    Eq. (12): L_{NCE}
    """
    def __init__(self, n_data):
        super(NCECriterion, self).__init__()
        self.n_data = n_data
        print('NCE approximation')

    def forward(self, x):
        bsz = x.shape[0]
        m = x.size(1) - 1

        # noise distribution
        Pn = 1 / float(self.n_data)

        # loss for positive pair
        P_pos = x.select(1, 0)
        log_D1 = torch.div(P_pos, P_pos.add(m * Pn + eps)).log_()

        # loss for K negative pair
        P_neg = x.narrow(1, 1, m)
        log_D0 = torch.div(P_neg.clone().fill_(m * Pn), P_neg.add(m * Pn + eps)).log_()
        # print(log_D0.size())
        loss = - (log_D1.sum(0) + log_D0.view(-1, 1).sum(0)) / bsz

        return loss

class NCESoftmaxLoss(nn.Module):
    """Softmax cross-entropy loss (a.k.a., info-NCE loss in CPC paper)"""
    def __init__(self):
        super(NCESoftmaxLoss, self).__init__()
        
        self.criterion = nn.CrossEntropyLoss()
        print('Using info_NCE based approximation')

    def forward(self, x):
        bsz = x.shape[0]
        # print(x.shape)
        x = x.squeeze()
        # print(x.shape)
        label = torch.zeros([bsz]).cuda().long()
        # label = torch.zeros([bsz]).long()
        loss = self.criterion(x, label)
        return loss

if __name__=='__main__':
    
    import numpy as np
    import random
    # localdim = LocalDim(in_channel=64, in_fc=128, out_fc=64)
    contrast = MemOpsLocDim(128, 100, 20, 0.07, 0.5).cuda()
    # nce = NCESoftmaxLoss()
    nce = NCECriterion(100)
    #K=20
    #total_samples=100
    #feat_dim=128


    #A batch sample
    z1 = torch.rand(32, 128)
    z2 = torch.rand(32, 128, 3, 3)
    y  = torch.from_numpy(np.asarray(random.sample(range(1, 100), 32)).reshape(32))
    # print(y.size())

    lx, lz = contrast(z1.to('cuda:0'), z2.to('cuda:0'), y.to('cuda:0'))
    print(lx.shape, lz.shape)
    # print(nce(lz))