
# from model import LocalEncoder, SelfAttention
import os
import datetime
import numpy as np
import pickle
import tensorboard_logger as tb_logger
from torch._C import device
import torch
import torchvision
from torchvision.utils import make_grid
from torch.utils import data
from torchvision import transforms, utils
from torch import nn
import matplotlib.pyplot as plt
import time
import torch.backends.cudnn as cudnn
from torchvision import utils
import sys
from utils import adjust_learning_rate, pil_loader
from loader import ImageData
from NCE.memory_bank import* 
from augment import*
import model_vgg
import model_res
# from model import*


import argparse


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
    parser.add_argument('--batch_size', type=int, default=10, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=0, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=250, help='number of training epochs')

    # optimization
    parser.add_argument('--optimizer_type', type=str, default='SGD', choices=['SGD', 'Adam'])
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='120', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    # parser.add_argument('--init', type=str, default='rand', choices= ['salgan', 'rand', 'vgg'], help='Initializing model')

    # resume path
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    # model type
    parser.add_argument('--model_type', default='vgg', type=str, choices=['vgg', 'resnet'])
    parser.add_argument('--encode_type', default='selfatt', type=str, choices=['dim', 'selfatt']) #self attention is complete model (includes dim)
    parser.add_argument('--init_type', default='random', type=str, choices=['salgan', 'random'])
    parser.add_argument('--lmda', default=0.7, type=float, help='ranges between 0 and 1')

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
    # opt.model_name = 'encode_{}_init_{}_memory_{}_{}_lr_{}_decay_{}_bsz_{}_optim_{}'.format(opt.encode_type, opt.init,opt.method, opt.nce_k, opt.learning_rate,
    #                                                                 opt.weight_decay, opt.batch_size, opt.optimizer_type)

    if opt.model_type=="resnet":
        # opt.encode_type='selfatt'
        opt.init_type='random'
        
    opt.model_name = 'model_{}_encode_{}_init_{}_lambda_{}_bsz_{}'.format(opt.model_type,
                                                                          opt.encode_type,
                                                                          opt.init_type,
                                                                          opt.lmda,
                                                                          opt.batch_size)

    if opt.amp:
        opt.model_name = '{}_amp_{}'.format(opt.model_name, opt.opt_level)

    opt.model_folder = os.path.join(opt.model_path,
                                    opt.model_type,
                                    opt.encode_type,
                                    opt.init_type,
                                    opt.model_name)

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
    

    if args.model_type=="resnet":

        if args.encode_type=="dim":
            net = model_res.LocalEncoder(args.device, args.feat_dim)
        elif args.encode_type=="selfatt":
            net = model_res.SelfAttLocDim(args.device, args.feat_dim)

    
    if args.model_type=="vgg":
        
        init=None
        if args.init_type=="salgan":
            init="salgan"
        if args.encode_type=="dim":
            net = model_vgg.LocalEncoder(args.device, args.feat_dim, init=init)
        elif args.encode_type=="selfatt":
            net = model_vgg.SelfAttLocDim(args.device, args.feat_dim, init=init)
    
    
    contrast =  MemOpsLocDim(args.feat_dim,\
                            n_data, args.nce_k,\
                            args.nce_t, args.nce_m,\
                            )

    nce_l1 = NCESoftmaxLoss() if args.softmax else NCECriterion(n_data)
    nce_l2 = NCESoftmaxLoss() if args.softmax else NCECriterion(n_data)
    
    
    
    #-------------------------
    net = net.to(args.device)
    
    nce_l1 = nce_l1
    nce_l2 = nce_l2
    
    if device !='cpu':
        cudnn.benchmark = True
    #======================================================================    

    return net, contrast, nce_l1, nce_l2
#-------------------------------------------------------------------------
def init_optimizer(args, net):
    
    
                  
    optimizer = torch.optim.SGD(net.parameters(),
                                    lr=args.learning_rate, 
                                    weight_decay=args.weight_decay)
    scheduler = 0

    return optimizer, scheduler

#-----------------------------------------------------------------------
def train(train_loader, net, contrast, nce_l1, nce_l2, optimizer, scheduler, args):
    
    
    net.train()
    contrast.train()

    steps = args.start_steps

    for epoch in range(args.start_epoch, args.epochs+1):

        #adjust learning rate
        if not args.model_type=="resnet":
            adjust_learning_rate(epoch, args, optimizer)
        

        t1 = time.time()
        
        
        for idx, (x, xt, index) in enumerate(train_loader):
            
            # print(idx)
            net.zero_grad()
              
            x, xt  = x.to(args.device), xt.to(args.device)  
            
            index = index.long().to(args.device)
            

            #-------forward---------------
            if args.encode_type=='selfatt':
                f, ft, gamma  = net(x, xt)
            else:
                f, ft= net(x, xt)
            #  print(f[0].m()) 
            #--------loss-----------------
            
            l1, l2 = contrast(f, ft, index)
            infonce_1, infonce_2 = nce_l1(l1), nce_l2(l2)

            #----------batch_loss---------
            #lambda*l1 + (1-lambda)*l2
            loss = (1.0-args.lmda)*infonce_1 + 1.0*(args.lmda)*infonce_2
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
                sys.stdout.flush()
            #logs
            if (steps%args.tb_freq==0):
                args.logger.log_value('total_loss', loss.item(), steps)
                args.logger.log_value('loss_nce1', infonce_1.item(), steps)
                args.logger.log_value('loss_nce2', infonce_2.item(), steps)
                if args.encode_type=="selfatt":
                    args.logger.log_value('Gamma', gamma.item(), steps)
                    
            
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
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    train_loader, n_data = get_data_loader(args)
   
       
    args.total_steps = args.epochs*int(n_data/args.batch_size)\
                      if (n_data%args.batch_size)==0\
                      else args.epochs*(int(n_data/args.batch_size)+1)  

    #get model
    net, contrast, nce_l1, nce_l2 = init_model(args,n_data)

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
