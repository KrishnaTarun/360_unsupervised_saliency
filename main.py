
import os
import datetime
import numpy as np
import pickle
import tensorboard_logger as tb_logger
from torch.utils.tensorboard import SummaryWriter
import torch
import torchvision
from torchvision.utils import make_grid
from torch.utils import data
from torchvision import transforms, utils
from torch import nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import torch.backends.cudnn as cudnn
from torchvision import utils
import sys
from utils import adjust_learning_rate, pil_loader
from loader import ImageData
from NCE.memory_bank import* 
from augment import*
# from model import*
from model_res import*
from cosine_annealing import CosineAnnealingWarmUpRestarts
import argparse
# from utils

try:
    from apex import amp, optimizers
except ImportError:
    # print
    pass

def parse_options():
    parser = argparse.ArgumentParser('argument for training')

    

    parser.add_argument('--print_freq', type=int, default=50, help='print frequency in steps')
    parser.add_argument('--tb_freq', type=int, default=50, help='tb frequency in steps')
    parser.add_argument('--save_freq', type=int, default=1, help='save frequency in epoch')
    parser.add_argument('--batch_size', type=int, default=25, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=0, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=250, help='number of training epochs')

    # optimization
    parser.add_argument('--optimizer_type', type=str, default='SGD', choices=['SGD', 'Adam'])
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='120, 200', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--init', type=str, default='rand', choices= ['salgan', 'rand', 'vgg'], help='Initializing model')

    # resume path
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    # encoder type
    parser.add_argument('--encode_type', default='Local', type=str, choices=['Vanilla', 'Local'])

    # model definition
    parser.add_argument('--softmax', action='store_true', help='using softmax contrastive loss rather than NCE')
    parser.add_argument('--nce_k', type=int, default=16000, help='# negative samples')
    parser.add_argument('--nce_t', type=float, default=0.07, help='temperature parameters')
    parser.add_argument('--nce_m', type=float, default=0.5, help='momentum for updates in memory bank')
    parser.add_argument('--feat_dim', type=int, default=512, help='dim of feat for inner product')
    parser.add_argument('--layer', type=int, default=6, help='output layer')

    # dataset
    parser.add_argument('--dataset', type=str, default='Train')

    # specify folder
    parser.add_argument('--data_folder', type=str, default="data", help='path to data')
    parser.add_argument('--model_path', type=str, default="models", help='path to save model')
    parser.add_argument('--tb_path', type=str, default="runs", help='path to tensorboard')


    # mixed precision setting
    parser.add_argument('--amp', action='store_true', help='using mixed precision')
    parser.add_argument('--opt_level', type=str, default='O1', choices=['O1', 'O2'])

    # data crop threshold
    # parser.add_argument('--crop_low', type=float, default=0.2, help='low area in crop')

    opt = parser.parse_args()

    if (opt.data_folder is None) or (opt.model_path is None) or (opt.tb_path is None):
        raise ValueError('one or more of the folders is None: data_folder | model_path | tb_path')

    iterations = opt.lr_decay_epochs.split(',')

    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.method = 'softmax' if opt.softmax else 'nce'
    opt.model_name = 'encode_{}_init_{}_memory_{}_{}_lr_{}_decay_{}_bsz_{}_optim_{}'.format(opt.encode_type, opt.init,opt.method, opt.nce_k, opt.learning_rate,
                                                                    opt.weight_decay, opt.batch_size, opt.optimizer_type)

    if opt.amp:
        opt.model_name = '{}_amp_{}'.format(opt.model_name, opt.opt_level)

    opt.model_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.model_folder):
        os.makedirs(opt.model_folder)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    if not os.path.isdir(opt.data_folder):
        raise ValueError('data path not exist: {}'.format(opt.data_folder))

    return opt


def get_data_loader(args):

    #path to image folders
    data_folder = os.path.join(args.data_folder, 'Train', 'image')

    
    train_dataset = ImageData(pil_loader, 
                             data_folder, 
                             transform=AugmentImage())

    # train loader
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)

    # num of samples
    n_data = len(train_dataset)
    print('number of samples: {}'.format(n_data))

    return train_loader, n_data
#------------------------------------------------------------------

def init_model(args, n_data):
    #set device
    #need to change in case of multiple-GPUS
    #=====================================================================
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    #-----------remove this--------
    #device = torch.device('cpu')
    if args.encode_type =='Local':
        # localdim = LocalDistractor(in_channel=512, in_fc=512, out_fc=512)
        # net = LocalEncoder(device, args.feat_dim)
        net = SelfAttLocDim(device, args.feat_dim)
 

    # contrast = MemOps(args.feat_dim,\
    #                         n_data, args.nce_k,\
    #                         args.nce_t, args.nce_m,\
    #                         )
    contrast =  MemOpsLocDim(args.feat_dim,\
                            n_data, args.nce_k,\
                            args.nce_t, args.nce_m,\
                            )

    nce_l1 = NCESoftmaxLoss() if args.softmax else NCECriterion(n_data)
    nce_l2 = NCESoftmaxLoss() if args.softmax else NCECriterion(n_data)
    
    # nce_l1 =  NCECriterion  
    
    #-------------------------
    net = net.to(device)
    # contrast = contrast.to(device)
    # nce_l1 = nce_l1.to(device)
    # nce_l2 = nce_l2.to(device)
    
    nce_l1 = nce_l1
    nce_l2 = nce_l2
    
    if device !='cpu':
        cudnn.benchmark = True
    #======================================================================    

    return net, contrast, nce_l1, nce_l2, device
#-------------------------------------------------------------------------
def init_optimizer(args, net):
    
    
                  
    optimizer = torch.optim.SGD(net.parameters(),
                                    lr=args.learning_rate, 
                                    weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate,
    #                               betas=(args.beta1, args.beta2),
    #                               weight_decay=args.weight_decay)

    # scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=50, T_mult=2, eta_max=args.learning_rate, T_up=10, gamma=0.5)
    scheduler = 0

    return optimizer, scheduler

#-----------------------------------------------------------------------
def train(train_loader, net, contrast, nce_l1, nce_l2, optimizer, scheduler, args):
    
    
    net.train()
    contrast.train()

    steps = args.start_steps

    for epoch in range(args.start_epoch, args.epochs+1):

        #adjust learning rate
        adjust_learning_rate(epoch, args, optimizer)
        # scheduler.step(epoch)
        # adjust_learning_rate(epoch, args, optimizer1)

        t1 = time.time()
        
        
        for idx, (x, xt, index) in enumerate(train_loader):
            
            # print(idx)
            net.zero_grad()
            
            
            # torchvision.utils.save_image(torchvision.utils.make_grid(z, nrow=10), 
            #                              fp='f.jpg')    
            
            x, xt  = x.cuda(), xt.cuda()  
            
            index = index.cuda().long()
            
            # input_ = torch.cat((x.unsqueeze(0), xt.unsqueeze(0)), dim=0)
            # print(input_.size(), input_.is_cuda)

            #-------forward---------------
            
            f, ft, gamma  = net(x, xt)
            
            #  print(f[0].m()) 
            #--------loss-----------------
            
            l1, l2 = contrast(f, ft, index)
            infonce_1, infonce_2 = nce_l1(l1), nce_l2(l2)

            #----------batch_loss---------
            #lambda*l1 + (1-lambda)*l2
            loss = (0.3)*infonce_1 + (0.7)*infonce_2
            # loss = infonce_2

            #-----------backward----------
            
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            steps+=1
            #-------------print and tensorboard logs ------------------
            if (steps%args.print_freq==0):
                print('Epoch: {}/{} || steps: {}/{} || total loss: {:.3f} || loss_nce1: {:.3f} || loss_nce2: {:.3f}'\
                    .format(epoch, args.epochs, steps, args.total_steps,
                    loss.item(), infonce_1.item(), infonce_2.item()))

                # print("max feat1: {}, feat2: {}, feat3: {}".format(feat1.max().item(), feat2.max().item(), feat3.max().item()))    
                # print(f.mean().item(), ft.mean().item())
                sys.stdout.flush()
            #logs
            if (steps%args.tb_freq==0):
                args.logger.log_value('total_loss', loss.item(), steps)
                args.logger.log_value('loss_nce1', infonce_1.item(), steps)
                args.logger.log_value('loss_nce2', infonce_2.item(), steps)
                args.logger.log_value('Gamma', gamma.item(), steps)
                
                #TODO
                
                # count=0
                # for i, m in enumerate(net.modules()):
                #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                #         # print(i, m.weight.data.size(),m.weight.grad.size())
                #         # writer.add_scalar('Gradient/layer_'+str(count), m.weight.grad.abs().mean(), steps)
                #         print(m.weight.grad)
                #         writer.add_histogram('hist_layer_'+str(count), m.weight.grad, steps) 
                #         count+=1
                    

            
           #------------------------------------------------------------- 
        t2 = time.time()
        print('epoch {}, total time {:.2f}s'.format(epoch, (t2 - t1)))
        #------train--completley-------------------
        

        #-------------------save net (after every epoch)-------------------------------   
        if(epoch%args.save_freq==0):
            print('==> Saving...')
            state = {
                'model': net.state_dict(),
                'contrast': contrast.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'steps':steps,
            }
            if args.amp:
                state['amp'] = amp.state_dict()

            save_file = os.path.join(args.model_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

            # help release GPU memory
            del state
        torch.cuda.empty_cache()


def main():

    #set parser
    args = parse_options()
    #data loader
    
    train_loader, n_data = get_data_loader(args)
   
       
    args.total_steps = args.epochs*int(n_data/args.batch_size)\
                      if (n_data%args.batch_size)==0\
                      else args.epochs*(int(n_data/args.batch_size)+1)  

    #get model
    net, contrast, nce_l1, nce_l2, args.device = init_model(args,n_data)

    #get optimizer
    optimizer, scheduler = init_optimizer(args, net)

    #mixed_precsion
    # if args.amp:
    #     model, optimizer = amp.initialize(net, optimizer, opt_level=args.opt_level)

    #checkpoint
    args.start_epoch = 1
    args.start_steps = 1
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch'] + 1
            args.start_steps  = checkpoint['steps'] + 1
            net.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            contrast.load_state_dict(checkpoint['contrast'])
            try:
                if args.amp and checkpoint['opt'].amp:
                    print('==> resuming amp state_dict')
                    amp.load_state_dict(checkpoint['amp'])
            except KeyError as e:
                print(str(e))
                pass
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint
            torch.cuda.empty_cache()
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    #tensorboard

    args.logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)
    # args.logger = SummaryWriter(log_dir=args.tb_folder)
    
    #start the training loop
    train(train_loader, net, contrast, nce_l1, nce_l2, optimizer, scheduler, args)
    
if __name__=='__main__':
    main()

# "-----------------------------------------------"
    # model_keys = [net.encoder.module.conv_block_1,
    #               net.encoder.module.conv_block_2,
    #               net.encoder.module.conv_block_3,
    #               net.encoder.module.conv_block_4,
    #               net.encoder.module.conv_block_5_1] 
    # copy_net_keys = []
    
    # # store keys
    # for k, v in model.items():
    #   # print("Layer {}".format(k))
      
    #   if 'encoder' in k.split('.'):
    #     copy_net_keys.append(k)
    # copy_net_keys = [tuple(copy_net_keys[i:i+2]) for i in range(0, len(copy_net_keys), 2)]
    
    # c = 0
    # for j, layer in enumerate(net.encoder.module.children()):
    #   try:
    #     # print(layer, j, model_keys[j][0].bias.shape)
        
    #     for id_, k in enumerate(layer.modules()):
    #       # print(id_)
    #       if isinstance(k, nn.Conv2d):
    #         with torch.no_grad():
    #           model_keys[j][id_-1].weight.copy_(model[copy_net_keys[c][0]])
    #           model_keys[j][id_-1].bias.copy_(model[copy_net_keys[c][1]])
    #         # print(c)
    #         c+=1
    
    #   except IndexError as e:
    #     pass
    # "------------------------------------------------------"