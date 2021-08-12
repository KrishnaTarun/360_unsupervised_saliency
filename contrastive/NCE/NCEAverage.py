import torch
from torch import nn
from alias_multinomial import AliasMethod
import math
import random
import numpy as np

class NCEAverage(nn.Module):

    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5, use_softmax=False):
        super(NCEAverage, self).__init__()
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        # self.multinomial.cuda()
        self.K = K
        self.use_softmax = use_softmax

        self.register_buffer('params', torch.tensor([K, T, -1, -1, -1, momentum])) 
        stdv = 1. / math.sqrt(inputSize / 3)

        #intialize with random samples
        #No need of memory Z1 at the moment
        # self.register_buffer('memory_z1', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_z2', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_z3', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))

    def forward(self, z1, z2, z3, y, idx=None):

        K = int(self.params[0].item())
        T = self.params[1].item()
        # Z_1 = self.params[2].item()
        Z_2 = self.params[2].item()
        Z_3 = self.params[3].item()

        momentum = self.params[4].item()
        batchSize = z1.size(0)
        outputSize = self.memory_z2.size(0)
        inputSize = self.memory_z2.size(1)

        # score computation
        if idx is None:
            idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1)
            idx.select(1, 0).copy_(y.data)
        # sample
        weight_z2 = torch.index_select(self.memory_z2, 0, idx.view(-1)).detach()
        weight_z2 = weight_z2.view(batchSize, K + 1, inputSize)
        I_z12 = torch.bmm(weight_z2, z1.view(batchSize, inputSize, 1))
        # sample
        weight_z3 = torch.index_select(self.memory_z3, 0, idx.view(-1)).detach()
        weight_z3 = weight_z3.view(batchSize, K + 1, inputSize)
        I_z13 = torch.bmm(weight_z3, z1.view(batchSize, inputSize, 1))

        if self.use_softmax:
            I_z12 = torch.div(I_z12, T)
            I_z13 = torch.div(I_z13, T)
            I_z12 = I_z12.contiguous()
            I_z13 = I_z13.contiguous()
        else:
            I_z12 = torch.exp(torch.div(I_z12, T))
            I_z13 = torch.exp(torch.div(I_z13, T))
            # set Z_0 if haven't been set yet,
            # Z_0 is used as a constant approximation of Z, to scale the probs
            if Z_2 < 0:
                self.params[2] = I_z12.mean() * outputSize
                Z_2 = self.params[2].clone().detach().item()
                print("normalization constant Z_2 is set to {:.1f}".format(Z_2))
            if Z_3 < 0:
                self.params[3] = I_z13.mean() * outputSize
                Z_3 = self.params[3].clone().detach().item()
                print("normalization constant Z_3 is set to {:.1f}".format(Z_3))
            # compute out_l, out_ab
            I_z12 = torch.div(I_z12, Z_2).contiguous()
            I_z13 = torch.div(I_z13, Z_3).contiguous()

        # # update memory
        with torch.no_grad():
            Z2_pos = torch.index_select(self.memory_z2, 0, y.view(-1))
            Z2_pos.mul_(momentum)
            Z2_pos.add_(torch.mul(z2, 1 - momentum))
            Z2_norm = Z2_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_z2 = Z2_pos.div(Z2_norm)
            self.memory_z2.index_copy_(0, y, updated_z2)

            Z3_pos = torch.index_select(self.memory_z3, 0, y.view(-1))
            Z3_pos.mul_(momentum)
            Z3_pos.add_(torch.mul(z3, 1 - momentum))
            Z3_norm = Z3_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_z3 = Z3_pos.div(Z3_norm)
            self.memory_z3.index_copy_(0, y, updated_z3)

        return I_z12, I_z13

if __name__=="__main__":

    contrast = NCEAverage(128, 100, 20, 0.07, 0.5)
    #K=20
    #total_samples=100
    #feat_dim=128


    #A batch sample
    z1 = torch.rand(32, 128)
    z2 = torch.rand(32, 128)
    z3 = torch.rand(32, 128)
    y  = torch.from_numpy(np.asarray(random.sample(range(1, 100), 32)).reshape(32))
    # print(y.size())

    _, _ = contrast(z1, z2, z3, y)



