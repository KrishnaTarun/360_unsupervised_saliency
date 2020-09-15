import torch
import torch.nn as nn
import torch.nn.functional as F
# from .alias_multinomial import AliasMethod
from .alias_multinomial import AliasMethod


class BaseMem(nn.Module):
    """Base Memory Class"""
    def __init__(self, K=65536, T=0.07, m=0.5):
        super(BaseMem, self).__init__()
        self.K = K
        self.T = T
        self.m = m

    def _update_memory(self, memory, x, y):
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
            updated_weight = F.normalize(w_pos)
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

    def __init__(self, n_dim, n_data, K=65536, T=0.07, m=0.5):
        super(MemOps, self).__init__(K, T, m)
        
        # create sampler
        self.multinomial = AliasMethod(torch.ones(n_data))
        self.multinomial.cuda()

        # create memory bank
        self.register_buffer('memory', torch.randn(n_data, n_dim))

        self.memory = F.normalize(self.memory)
        
        self.loss = NCESoftmaxLoss() 
    
    def forward(self, x, z, y):
        """
            x: untransformed image representation
            z: projected image representation
            y: index of x from Data-set

            L = (1-lambda)*L_nce(m, x) + (lambda)*L_nce(m, z)
        """
        bsz = x.shape[0]
        n_dim = x.size(1)
        
        #get representations from memory for x
        mI = torch.index_select(self.memory,0, y)
        
        # print(y.shape, y.view(-1).shape)
        # print(self.memory.shape, mI.shape) 
        
        #samples indexes (Negative K samples)
        idx = self.multinomial.draw(bsz * (self.K)).view(bsz, -1)
        
        #get representations for negative samples
        w = torch.index_select(self.memory, 0 , idx.view(-1))
        w = w.view(bsz, self.K, n_dim)

        #combine mI and w
        mI = torch.cat((mI.unsqueeze(1),w), dim=1)

        #compute logit for x and z
        lx = self._compute_logit(x, mI)
        lz = self._compute_logit(z, mI)


        #update memory with current feature of x
        self._update_memory(self.memory, x, y) 

        return lx, lz
        




class NCESoftmaxLoss(nn.Module):
    """Softmax cross-entropy loss (a.k.a., info-NCE loss in CPC paper)"""
    def __init__(self):
        super(NCESoftmaxLoss, self).__init__()
        
        self.criterion = nn.CrossEntropyLoss()

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
    contrast = MemOps(128, 100, 20, 0.07, 0.5)
    nce = NCESoftmaxLoss()
    #K=20
    #total_samples=100
    #feat_dim=128


    #A batch sample
    z1 = torch.rand(32, 128)
    z2 = torch.rand(32, 128)
    y  = torch.from_numpy(np.asarray(random.sample(range(1, 100), 32)).reshape(32))
    # print(y.size())

    lx, lz = contrast(z1, z2, y)
    print(lx.shape, lz.shape)
    print(nce(lx), nce(lz))