from functools import partial
import numpy as np
from numpy import random
import time
from skimage import exposure
from skimage.transform import resize
import cv2
from model import SalGAN, LOSS
from salgan import SalGAN_Generator
from model_res import VGG
from attention import Sal_based_Attention_module
import re, os, glob
import cv2
import torch
import scipy.misc
from torchvision import utils
from PIL import Image
import tensorboard_logger as tb_logger



model = SalGAN()
weight = torch.load('./encoder/ckpt_epoch_219.pth')
model_dict =  model.state_dict()
pretrained_dict = {"encoder_salgan."+'.'.join(k.split('encoder1')[-1].split('.')[1:]):
                        v for k, v in weight['model'].items() if 'encoder1' in k.split('.')}
model_dict.update(pretrained_dict)
model.load_state_dict(pretrained_dict, strict=False)
model.cuda()


model1 = SalGAN_Generator()
weight1 = torch.load('./salgan_3dconv_module.pt')

model1.load_state_dict(weight1, strict=False)
model1.cuda()




model2 = SalGAN_Generator()

weight2 = torch.load('./encoder/self_dim_250.pth')

model2.load_state_dict(weight2, strict=False)
model2.cuda()


#IMG_Path = "/home/yasser/Desktop/360 project/ftp.ivc.polytech.univ-nantes.fr/images/"
IMG_Path = "./dataset/test/images/"
image_list = os.listdir(IMG_Path)


if not os.path.isdir('./resnet'):
    os.makedirs('./resnet')

if not os.path.isdir('./vgg'):
    os.makedirs('./vgg')

print(image_list)
# i =80
total1 = []
total2 = []
total3 = []

for img in image_list:
    image_path = IMG_Path + img
    print(image_path)
    inpt = cv2.imread(image_path)
    inpt = cv2.resize(inpt,(320, 160))

    inpt = np.float32(inpt)
    inpt-=[0.485, 0.456, 0.406]
    inpt = torch.cuda.FloatTensor(inpt)

    inpt = inpt.permute(2, 0, 1)
    

    with torch.no_grad():
        latent1 = model(inpt.unsqueeze(0))

        latent2 = model1(inpt.unsqueeze(0))

        latent3 = model2(inpt.unsqueeze(0))
        

    latent1 = (latent1.cpu()).detach().numpy()
    latent1 = latent1.squeeze()
    latent1 = np.reshape(latent1,512*10*20)

    latent2 = (latent2.cpu()).detach().numpy()
    latent2 = latent2.squeeze()
    latent2 = np.reshape(latent2,512*10*20)

    latent3 = (latent3.cpu()).detach().numpy()
    latent3 = latent3.squeeze()
    latent3 = np.reshape(latent3,512*10*20)

    total1.append(latent1)
    total2.append(latent2)
    total3.append(latent3)

    """
    mse = (np.square(np.abs(latent1 - latent2))).mean(axis=0)
    print('MSE IS:  ',mse)
    total = total+mse
    
    np.save('./resnet/'+img[:-4]+'.npy',latent1)
    np.save('./vgg/'+img[:-4]+'.npy',latent2)
    """

total1 = np.array(total1)
total2 = np.array(total2)
total3 = np.array(total3)

np.save('total1.npy',total1)
np.save('total2.npy',total2)
np.save('total3.npy',total3)



