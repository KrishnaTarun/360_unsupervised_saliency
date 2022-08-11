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
from model_2 import*
from model_2 import SalGAN_2, LOSS




mean = lambda x: sum(x) / len(x)

def train(train_loader, model, criterion, optimizer, epoch, state='model'):
    # Switch to train mode
    model.train()

    print("Now commencing {} epoch {}".format(state, epoch))

    losses = []
    for j, batch in enumerate(train_loader):
        # print(batch[0].size())
        start = datetime.datetime.now().replace(microsecond=0)
        loss = torch.tensor(0)
        frame, gtruth = batch
        for i in tqdm(range(frame.size()[0])):
            # try:
            saliency_map = model(frame[i])
            # print(saliency_map.size())
            
            saliency_map = saliency_map.squeeze(0)
            # print(saliency_map.size(), gtruth[i].size())    
            total_loss = criterion(saliency_map, gtruth[i])
        
            loss = loss.item() + total_loss
            # except:
            # print('error')
            # continue
            
            if j % 5 == 0 and i == 3:

                post_process_saliency_map = (saliency_map - torch.min(saliency_map)) / (
                            torch.max(saliency_map) - torch.min(saliency_map))
                utils.save_image(post_process_saliency_map,
                                 "./map/smap{}_batch{}_epoch{}_{}.png".format(i, j, epoch, state))
                #if epoch == 0:
                utils.save_image(gtruth[0], "./map/{}_batch{}_grounftruth.png".format(i, j))
            
        loss.backward()
        optimizer.step()
        end = datetime.datetime.now().replace(microsecond=0)
        print('\n\tEpoch: {}\on {}\t Batch: {}\t Training Loss: {}\t Time elapsed: {}\t'.format(epoch, state, j,
                                                                                                loss.data /
                                                                                                frame.size()[0],
                                                                                                end - start))
        losses.append(loss.data / frame.size()[0])
    return (mean(losses))


def validate(val_loader, model, criterion, epoch):
    # Switch to train mode
    model.eval()

    losses = []
    
    for j, batch in enumerate(val_loader):
        # print(batch.size())
        print("load batch....")
        start = datetime.datetime.now().replace(microsecond=0)
        loss = 0
        frame, gtruth = batch
        for i in range(frame.size()[0]):
            #inpt = frame[i].unsqueeze(0).cuda()
            # print(frame.size())
            # torchvision.utils.save_image(torchvision.utils.make_grid(frame, nrow=10), fp='f.jpg') 
            with torch.no_grad():
                
                saliency_map = model(frame[i])
                saliency_map = saliency_map.squeeze(0)
                # post_process_saliency_map = (saliency_map - torch.min(saliency_map)) / (
                #             torch.max(saliency_map) - torch.min(saliency_map))
                # utils.save_image(post_process_saliency_map,
                #                 "./map/smap{}_batch{}.png".format(i, j))
                total_loss = criterion(saliency_map, gtruth[i])

            # print("last loss ",last.data)          Multiply()

            # print("attention loss ",attention.data)
            loss = loss + total_loss.data
            # loss = loss + attention
        end = datetime.datetime.now().replace(microsecond=0)
        print('\n\tEpoch: {}\Batch: {}\t Validation Loss: {}\t Time elapsed: {}\t'.format(epoch, j,
                                                                                          loss.data / frame.size()[0],
                                                                                          end - start))
        losses.append(loss.data / frame.size()[0])

    return (mean(losses))


def adjust_learning_rate(optimizer, learning_rate, epoch, decay_rate=0.1):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 11:
        lr = learning_rate * (decay_rate ** (epoch // 10))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


model = SalGAN_2()



val_perc = 0.1401

train_set = Static_dataset(
    # root_path='./dataset/training_set/',
    root_path='./dataset/train',
    load_gt=True,
    resolution=(320, 160),
    split="train")
print("Size of train set is {}".format(len(train_set)))
train_loader = data.DataLoader(train_set, batch_size=512, shuffle=True, num_workers=0, drop_last=True)
valid_set = Static_dataset(
    root_path='./dataset/test/',
    load_gt=True,
    resolution=(320, 160),
    split="validation")

print("Size of validation set is {}".format(len(valid_set)))
val_loader = data.DataLoader(valid_set, batch_size=128 , num_workers=0, drop_last=True)
criterion = nn.BCELoss()

# weight initialization 
#-------------------------------------------------------------
#Load the trained stuff for encoder and salgan for decoder

optimizer = torch.optim.Adam([
    {'params': model.parameters(), 'lr': 0.0001, 'weight_decay': 0.00001}])

"""
#------override encoder weights from unsupervised model-------
weight = torch.load('./encoder/ckpt_epoch_219.pth')
# weight = torch.load('/home/tarun/Documents/Phd/360_cvpr/models/init_rand_memory_nce_16000_lr_0.01_decay_0.0001_bsz_25_optim_SGD/ckpt_epoch_138.pth')



model_dict =  model.state_dict()
# print(len(model_dict.keys()))
# print(model_dict.keys())
pretrained_dict = {"encoder_salgan."+'.'.join(k.split('encoder1')[-1].split('.')[1:]):
                        v for k, v in weight['model'].items() if 'encoder1' in k.split('.')}
model_dict.update(pretrained_dict)
model.load_state_dict(pretrained_dict, strict=False)


#model.encoder_salgan.load_state_dict(weight['model'], strict=False)
del weight
#model.encoder_salgan.apply(init_weights)

# model_dict = model.state_dict()
# pretrained_dict = {k: v for k, v in weight['model'].items() if 'encoder' in k.split('.')}
# key_pre = list(pretrained_dict.keys())
# i=0
# del weight
# print('=====================================')
# for k, v in model_dict.items():
#     # print(k)
#     if 'encoder' in k.split('.'):
#         print(k,'----->',key_pre[i])
#         model_dict[k] = pretrained_dict[key_pre[i]]
#         i=1+i
# model.load_state_dict(model_dict)
# print('=====================================')
# ----------------freeze encoder------------------------

for id_, k in enumerate(model.encoder_salgan.modules()):

    if isinstance(k, nn.Conv2d):
        k.weight.requires_grad = False
        try:
            k.bias.requires_grad = False
        except:
            pass

"""
model.cuda()

# --------------------------------------------------------------


cudnn.benchmark = True
criterion = criterion.cuda()
# Traning #
starting_time = datetime.datetime.now().replace(microsecond=0)
print("Training started at : {}".format(starting_time))
train_losses = []
val_losses = []
start_epoch = 0
epochs = 1000
plot_every = 1



# val_loss = validate(val_loader, model, criterion, 1)
for epoch in tqdm(range(epochs)):
    for epoch in range(start_epoch, epochs+1):

        print('**** new epoch ****')
        # train for one epoch
        train_loss = train(train_loader, model, criterion, optimizer, epoch)

        print("Epoch {}/{} done with train loss {}\n".format(epoch, epochs, train_loss))

        if val_perc > 0:
            print("Running validation..")
            val_loss = validate(val_loader, model, criterion, epoch)
            print("Validation loss: {}\t  ".format(val_loss))

        if epoch % plot_every == 0:
            train_losses.append(train_loss.cpu())
            if val_perc > 0:
                val_losses.append(val_loss.cpu())
        print("\n epoch finished at : {} \n Now saving..".format(datetime.datetime.now().replace(microsecond=0)))

        for param_group in optimizer.param_groups:
            print(param_group['lr'])
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.cpu().state_dict(),
            'optimizer1_state_dict': optimizer.state_dict(),
        }, "./weights/model{}.pt".format(epoch))

        model = model.cuda()
        to_plot = {
            'epoch_ticks': list(range(start_epoch, epoch + 1, plot_every)),
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        with open('./train_plot.pkl', 'wb') as handle:
            pickle.dump(to_plot, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Training of new model started at {} and finished at : {} \n ".format(starting_time,
                                                                                datetime.datetime.now().replace(
                                                                                    microsecond=0)))
