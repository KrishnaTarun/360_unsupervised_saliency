import cv2
import os
import datetime
import numpy as np
import pickle
import torch
from torch.utils import data
from torchvision import transforms, utils
from torch import nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from data_loader import Static_dataset
import argparse
from model import*
from model import SalGAN, LOSS
from model_2 import*
from model_2 import SalGAN_4, LOSS,SalGAN_3
import tensorboard_logger as tb_logger
import torchvision
import time
import sys
sys.path.insert(0, '/home/tarun/Documents/Phd/360_cvpr/evaluate/')

mean = lambda x: sum(x) / len(x)
#TODO adjust lerning rate if required
#TODO weight load
#TODO checkpoint
#TODO write loss for batch
#TODO check nn.DataParallel

def parse_options():
    
    parser = argparse.ArgumentParser('argument for training')

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10, help='print frequency in steps')
    parser.add_argument('--tb_freq', type=int, default=10, help='tb frequency in steps')
    parser.add_argument('--save_freq', type=int, default=1, help='save frequency in epoch')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=0, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')

    #optimization
    parser.add_argument('--optimizer_type', type=str, default='Adam', choices=['SGD', 'Adam'])
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='50,100', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    
    #evaluation_training_procedure
    parser.add_argument('--run_type', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--encoder', type=str,  default='resnet_pool', choices=['salgan', 'un','resnet','rand'],
                                                                help='encoder initialization')
    parser.add_argument('--decoder', type=str,  default='rand', choices=['salgan', 'rand'],
                                                                help='decoder initialization')
    parser.add_argument('--salgan_path', type=str, default= "./")
    parser.add_argument('--unsup_path', type=str, default="./encoder/pooling_ckpt_epoch_250.pth")

    
    #resume path
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')


    # dataset
    parser.add_argument('--dataset', type=str, default='train')

    # specify folder
    parser.add_argument('--data_folder', type=str, default="./dataset", help='path to data')
    parser.add_argument('--model_path', type=str, default="models", help='path to save model')
    parser.add_argument('--tb_path', type=str, default="runs", help='path to tensorboard')


    opt = parser.parse_args()
    if (opt.data_folder is None) or (opt.model_path is None) or (opt.tb_path is None):
        raise ValueError('one or more of the folders is None: data_folder | model_path | tb_path')

    iterations = opt.lr_decay_epochs.split(',')

    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    
    opt.model_name = 'model_encoder:_{}_decoder:_{}_lr:_{}_bz:_{}_opt:_{}'.format(opt.encoder, opt.decoder, opt.learning_rate,
                                                                                opt.batch_size, opt.optimizer_type)
    opt.result_folder = os.path.join('result', opt.run_type, opt.model_name)

    opt.model_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.model_folder):
        os.makedirs(opt.model_folder)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)
    
    if not os.path.isdir(opt.result_folder):
        os.makedirs(opt.result_folder)
    # if os.path.isdir('dataset'):
    #     print('hell')
    if not os.path.isdir(opt.data_folder):
        raise ValueError('data path not exist: {}'.format(opt.data_folder))

    return opt

#TODO load_data

def get_data_loader(args):
    
    train_set = Static_dataset(
                    root_path=os.path.join(args.data_folder, args.dataset),
                    resolution=(320, 160), #(w, h)
                    split=args.run_type
    )

    n_data = len(train_set)
    print('Number of training samples {}'.format(n_data))
    train_loader = data.DataLoader(train_set,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=0,
                                   drop_last=True)
    valid_set = Static_dataset(
                    root_path=os.path.join(args.data_folder, 'test'),
                    resolution=(320, 160), #(w, h)
                    split=args.run_type
                    
    )
    valid_loader  = data.DataLoader(valid_set,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=0,
                                   drop_last=True)
    print('Number of validation samples {}'.format(len(valid_set)))                                                   

    return train_loader, n_data, valid_loader

def validation(valid_loader, net, epoch, criterion, args):

    net.eval()
    criterion.eval()

    
    print('---------------Validating----------------')
    for idx, (im, sal_gt,fix_gt) in enumerate(valid_loader):

        total_loss =[]
                                    
        im, sal_gt,fix_gt  = im.cuda(), sal_gt.cuda(),fix_gt.cuda()
        with torch.no_grad():
            sal_pt = net(im)
            loss = criterion(sal_pt, sal_gt,fix_gt)
            total_loss.append(loss.data)
    args.logger.log_value('val_loss_per_batch', mean(total_loss).item()/64.0, epoch)
    print('Validation Loss: {}'.format(mean(total_loss).item()/64.0))
        


def init_model(args, n_data):

    
    def init_weights(m):
        if type(m) == nn.Conv2d:
            # print('yes')
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    model = SalGAN_4()
    
    #-sal_weight = torch.load(args.salgan_path)
    criterion = LOSS()
    
    #model.decoder_salgan.load_state_dict(sal_weight, strict=False)    
    #--------encoder----------------
    if args.encoder=='salgan':
        # model.encoder_salgan.load_state_dict(sal_weight, strict=False)
        pass
    
    if args.encoder=='un':   
        un_weight  = torch.load(args.unsup_path)
        model.encoder_salgan.load_state_dict(un_weight['model'], strict=False)     
        del un_weight
    

    if args.encoder=='resnet_pool':

        res_weight  = torch.load(args.unsup_path)
        model_dict =  model.state_dict()
        pretrained_dict = {"encoder_salgan."+'.'.join(k.split('encoder1')[-1].split('.')[1:]):
                            v for k, v in res_weight['model'].items() if 'encoder1' in k.split('.')}
        model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
    
    # #----------decoder---------------
    if args.decoder=='salgan':
        # model.decoder_salgan.load_state_dict(sal_weight, strict=False)
        pass

    if args.decoder=='rand':
        model.decoder_salgan.apply(init_weights)
    
    #-del sal_weight
    #-----------------------------------
    print('==============================================================')        
    print('Train model with encoder initialized with {}\
                            and decoder with {}'.format(args.encoder,\
                                                        args.decoder))
    print('==============================================================')
    #-----------freeze encoder----------
    
    for id_, k in enumerate(model.encoder_salgan.modules()):
        if isinstance(k, nn.Conv2d):
            k.weight.requires_grad = False
            """
            try:
                k.bias.requires_grad = False
            except:
                pass   
            """
    
    if torch.cuda.is_available():
        model.cuda()
        criterion.cuda()
        #model.to('cuda:1')
        #criterion.to('cuda:1')
        cudnn.benchmark=True
    
    return model.train(), criterion.train() 
    
    # if args.run_type=='test':
        
    #     #load the trained model to test on your data
    #     weight = torch.load(args.model_path)
    #     model.load_state_dict(weight, strict=False)
    #     del weight

    #     return model.eval(), criterion




def init_optimizer(args, model):

    optimizer = torch.optim.Adam(
                            params=model.parameters(),
                            betas=(args.beta1, args.beta2),
                            lr=args.learning_rate,
                            weight_decay=args.weight_decay)

    return optimizer                                                   

#TODO train
def train(train_loader, valid_loader, net,  criterion, optimizer, args):
    
    steps = args.start_steps
    start_time = time.time()
    
    for epoch in range(args.start_epoch, args.epochs+1):

        #adjust learning rate
        #TODO
        # adjust_learning_rate(epoch, args, optimizer)
        

        t1 = time.time()
        
        net.train()
        criterion.train()    
        for idx, (im, sal_gt,fix_gt) in enumerate(train_loader):
            
            
            net.zero_grad()
            im, sal_gt,fix_gt  = im.cuda(), sal_gt.cuda(),fix_gt.cuda()
            
            # im, sal_gt  = im.to('cuda:1'), sal_gt.to('cuda:1')
            
            

            #-------forward---------------
            
            sal_pt = net(im)
            loss = criterion(sal_pt, sal_gt,fix_gt)
            #-------backward---------------
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            post_sal = sal_pt[:5, :,:,:]
            # min_val = torch.min(post_sal.view(5,-1), dim=1, keepdim=True).values
            # max_val = torch.max(post_sal.view(5,-1), dim=1, keepdim=True).values
            
            if (steps%args.print_freq==0):
                post_sal = sal_pt[:5, :,:,:]
                min_val = torch.min(post_sal.view(5,-1), dim=1, keepdim=True).values
                max_val = torch.max(post_sal.view(5,-1), dim=1, keepdim=True).values
                min_val = min_val.view(5,1,1,1)
                max_val = max_val.view(5,1,1,1)
                
                post_pro_sal = (post_sal - min_val)/(max_val-min_val) 
                z = torch.cat((sal_gt[0:5,:,:,:], post_pro_sal), dim=0)
                torchvision.utils.save_image(torchvision.utils.make_grid(z, nrow=5),
                                              fp=os.path.join(args.result_folder, 
                                                            args.run_type + '_epoch_:' +
                                                            str(epoch) + '_steps_:' +
                                                            str(steps)+'.png'))
                print('Epoch: {}/{} || steps: {}/{} || total loss: {:.3f}'\
                                                .format(epoch, args.epochs,
                                                steps,
                                                args.total_steps,
                                                loss.item()/64.0))  
            
                
                 
            
            steps+=1

        t2 = time.time()
        args.logger.log_value('Train_loss', loss.item()/64.0, steps)
        print('epoch {}, total time {:.2f}s'.format(epoch, (t2 - t1)))
        #------train--completley-------------------
        

        #-------------------save net (after every epoch)-------------------------------   
        if(epoch%args.save_freq==0):
            print('==> Saving...')
            state = {
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'steps':steps,
            }
            save_file = os.path.join(args.model_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

            # help release GPU memory
            del state
        validation(valid_loader, net, epoch, criterion, args)
        torch.cuda.empty_cache()

            # TODO run validation
    print('----------------------------------------------')
    print('training_took {}'.format(time.time()-start_time))

def main():

    #set parser
    args = parse_options()
    #data loader
    
    train_loader, n_data, valid_loader = get_data_loader(args)
   
       
    args.total_steps = args.epochs*int(n_data/args.batch_size)\
                      if (n_data%args.batch_size)==0\
                      else args.epochs*(int(n_data/args.batch_size)+1)  

    #get model
    net, criterion = init_model(args,n_data)

    #get optimizer
    optimizer = init_optimizer(args, net)


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
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint
            torch.cuda.empty_cache()
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    #tensorboard
    args.logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)
    
    #start the training loop
    train(train_loader, valid_loader, net, criterion, optimizer, args)
    
if __name__=='__main__':
    main()


