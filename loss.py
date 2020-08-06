def _pointwise_loss(lambd, input, target, size_average=True, reduce=True):
    d = lambd(input, target)
    if not reduce:
        return d
    return torch.mean(d) if size_average else torch.sum(d)


class KLDLoss(nn.Module):
    def __init__(self):
        super(KLDLoss, self).__init__()

    def KLD(self, inp, trg):
        inp = inp / torch.sum(inp)
        trg = trg / torch.sum(trg)
        eps = sys.float_info.epsilon

        return torch.sum(trg * torch.log(eps + torch.div(trg, (inp + eps))))

    def forward(self, inp, trg):
        return _pointwise_loss(lambda a, b: self.KLD(a, b), inp, trg)
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

class LOSS(nn.Module):
    def __init__(self):
        super(LOSS, self).__init__()
        self.KLDLoss = KLDLoss()
        self.CC_Loss=CC_Loss()
        self.NSSLoss=NSSLoss()


    def forward(self, saliency_map, gtruth):
        loss = self.KLDLoss(saliency_map, gtruth)-0.1*self.CC_Loss(saliency_map, gtruth)

        return loss