from functools import partial
import numpy as np
from numpy import random
import time
from skimage import exposure
from skimage.transform import resize
import cv2
from model_2 import SalGAN_3, LOSS,SalGAN_4,SalGAN_2
import re, os, glob
import cv2
import torch
import scipy.misc
from torchvision import utils
from PIL import Image
from attention import Sal_based_Attention_module
from salgan import SalGAN_Generator


def normalize(x, method='standard', axis=None):
    x = np.array(x, copy=False)
    if axis is not None:
        y = np.rollaxis(x, axis).reshape([x.shape[axis], -1])
        shape = np.ones(len(x.shape))
        shape[axis] = x.shape[axis]
        if method == 'standard':
            res = (x - np.mean(y, axis=1).reshape(shape)) / np.std(y, axis=1).reshape(shape)
        elif method == 'range':
            res = (x - np.min(y, axis=1).reshape(shape)) / (np.max(y, axis=1) - np.min(y, axis=1)).reshape(shape)
        elif method == 'sum':
            res = x / np.float_(np.sum(y, axis=1).reshape(shape))
        else:
            raise ValueError('method not in {"standard", "range", "sum"}')
    else:
        if method == 'standard':
            res = (x - np.mean(x)) / np.std(x)
        elif method == 'range':
            res = (x - np.min(x)) / (np.max(x) - np.min(x))
        elif method == 'sum':
            res = x / float(np.sum(x))
        else:
            raise ValueError('method not in {"standard", "range", "sum"}')
    return res

def CC(saliency_map1, saliency_map2):
	map1 = np.array(saliency_map1, copy=False)
	map2 = np.array(saliency_map2, copy=False)
	if map1.shape != map2.shape:
		map1 = resize(map1, map2.shape, order=3, mode='constant')
	map1 = normalize(map1, method='standard')
	map2 = normalize(map2, method='standard')
	return np.corrcoef(map1.ravel(), map2.ravel())[0,1]



#Contrastive

model = SalGAN_4()

weight = torch.load('./models/model_encoder:_resnet_pool_decoder:_rand_lr:_0.0001_bz:_64_opt:_Adam/ckpt_epoch_56.pth')
#weight = torch.load('./trained_models/random_decoder/final_models/finetuned.pth')

model.load_state_dict(weight['model'], strict=True)


"""
#Salgan
model = SalGAN_Generator()
weight = torch.load('./salgan_3dconv_module.pt')

model.load_state_dict(weight, strict=False)

"""

"""
#Attention
model = Sal_based_Attention_module()

weight = torch.load('./attention.pt')
model.load_state_dict(weight['state_dict'], strict=False)

"""


model.cuda()


IMG_Path = "/home/yasser/Desktop/DATA360/images/"
#IMG_Path = "/home/yasser/Desktop/Paper_figure/images_full/"

#IMG_Path = "./sitzman/Test/image/"

#IMG_Path = "./dataset/test/images/"

image_list = os.listdir(IMG_Path)


if not os.path.isdir('./V'):
    os.makedirs('./V')

if not os.path.isdir('./saliency'):
    os.makedirs('./saliency')


print(image_list)
# i =80
for img in image_list:
    image_path = IMG_Path + img
    print(img)
    ori = cv2.imread(image_path)
    inpt = cv2.resize(ori,(320, 160))

    inpt = np.float32(inpt)
    #inpt-=[0.485, 0.456, 0.406]
    inpt = torch.cuda.FloatTensor(inpt)

    inpt = inpt.permute(2, 0, 1)
    
    inpt = torch.cuda.FloatTensor(inpt)

    with torch.no_grad():
        saliency_map = model(inpt.unsqueeze(0))

    Output = saliency_map
    Output = (Output.cpu()).detach().numpy()
    Output = Output.squeeze()
    Output = resize(Output, (1024,2048))
    np.save('./V/'+img[:-4]+'.npy',Output)
    cv2.imwrite('./saliency/'+img[:-4]+'.png',(Output-Output.min())*255/(Output.max()-Output.min()))

"""
EPSILON = np.finfo('float').eps

import numpy as np
import os
vggnet = os.listdir('./R/')
vgg = []
for i in vggnet:
    z = np.load('./R/'+i)
    vgg.append(z)
vgg = np.array(vgg)
vgg.shape
gl =[]
sample = []
score =0
for i in range(0,26):
     zi = vgg[i]
     gl.append(sample)
     sample = []
     for j in range(0,512):
             zj = zi[j]
             #zj = np.reshape(zj,zj.shape[0]*zj.shape[1])
             sample.append(np.absolute(score))
             for m in range(0,512):
                     zs = zi[m]
                     #zs = np.reshape(zs,zs.shape[0]*zs.shape[1])
                     zs = zs / np.linalg.norm(zs)
                     zj = zj / np.linalg.norm(zj)
                     #score = (np.square(np.absolute(zj-zs))).mean()
                     score = CC(zj,zs)
                     print(np.absolute(score))

                    
s=[]
for i in range(1,25):
     b = gl[i]
     b = np.array(b)
     b = b[np.logical_not(np.isnan(b))]
     s.append(b.mean())
s= np.array(s)
s.mean()
print(str(s))


import numpy as np
DATA = []
for i in range(1,110):
    z = np.load('./'+str(i)+'.npy',allow_pickle=True)
    DATA.append(z)
Data = np.array(DATA,dtype=object)   
indices = []
for i in range(0,Data[0].shape[0]):
    indices.append(Data[0][i][1][0])
for ind in indices:
    z = []
    for i in range(0,105):
        for j in range(0,Data[i].shape[0]):
            if Data[i][j][1][0] == ind:
                z.append(Data[i][j][0])
    z = np.array(z)
    np.save('./indices/'+str(ind)+'.npy',z)

for i in lis:
     z = np.load('./'+i)
     res = []
     for j in range(0,z.shape[0]-1):
             score = CC(z[j],z[j+1])
             print(score)
             res.append(score)
     res = np.array(res)
     np.save('./scores/'+str(i),res)

import matplotlib.pyplot as plt
import numpy as np
import os
lis = os.listdir('./')
mean = []
for i in lis:
    z = np.load('./'+i) 
    mean.append(z.mean())
mean
mean = np.array(mean)
mean.mean()
for i in lis[0:4]:
    z = np.load('./'+i) 
    plt.plot(z)

import numpy as np
import os
ind = np.load('./0.2_diff_labels.npy')
l = np.delete(ind, np.where(ind>49664))
l = np.random.choice( l,1000, replace=False)
latent = []
for i in range(1,108):
     print(str(i))
     z = np.load('./'+str(i)+'.npy',allow_pickle=True)
     z = np.reshape(z,(97*512,2))
     z = z.T
     r = []
     for j in range(0,1000):
             ind = l[j]
             for m in range(0,49664):
                     if z[0][m] == ind:
                             r.append(z[1][m])
     r = np.array(r)
     latent.append(r)

from scipy.spatial import distance
C =[]
N =[]
z1_N = noisy_z[0]
z1_C = clean_z[0]
for i in range(4,65):
    z1_N = noisy_z[i]
    #z2_N = noisy_z[i+5]
    z1_C = clean_z[i]
    #z2_C = clean_z[i+5]
    NN = []
    CC = []
    for j in range(0,200):
        zs = z1_N[j]
        zj = z2_N[j]
        zs = zs / np.linalg.norm(zs)
        zj = zj / np.linalg.norm(zj)
        zss = z1_C[j]
        zjj = z2_C[j]
        zss = zss / np.linalg.norm(zss)
        zjj = zjj / np.linalg.norm(zjj)
        score_noisy = 1-distance.cosine(zs,zj)
        score_clean = 1-distance.cosine(zss,zjj)
        NN.append(score_noisy)
        CC.append(score_clean)
    NN = np.array(NN)
    CC = np.array(CC)
    C.append(CC)
    N.append(NN)

C = np.array(C)
N = np.array(N)
CC = []
NN = []
for i in range(0,N.shape[0]):
    CC.append(C[i].mean())
    NN.append(N[i].mean())


NN = np.array(NN)
CC = np.array(CC)
plt.plot(NN, 'g',label='NOISY')
plt.plot(CC, 'b',label='CLEAN')
plt.xlabel('Epochs')
plt.ylabel('Cosine distance')
plt.legend()
plt.show()
import numpy as np
data = []
for i in range(35,100):
    z = np.load('./'+str(i)+'.npy',allow_pickle=True)
    z = np.reshape(z,(97*512,2))
    data.append(z)
data=np.array(data)
indices = data.T[0]
noisy_index = np.load('./0.2_diff_labels.npy')
clean_indexes=np.random.randint(50000, size=20000)
mask = np.isin(clean_indexes,noisy_index)
mask = np.logical_not(mask)
clean_indexes = clean_indexes[mask]
m = np.where(clean_indexes == noisy_ref)
m = np.isin(clean_indexes,noisy_ref)
clean_indexes = np.random.choice( clean_indexes,1000, replace=False)
clean_z = []
noisy_z = []
for i in range(0,65):
    clean_lat = []
    noisy_lat = []
    for j in range(0,1000):
        noise = noisy_index[j]
        clean = clean_indexes[j]
        for m in range(0,49664):
            if data[i][m][0] == noise:
                noisy_lat.append(data[i][m][1])
            if data[i][m][0] == clean:
                clean_lat.append(data[i][m][1])
    clean_lat = np.array(clean_lat)
    noisy_lat = np.array(noisy_lat)
    clean_z.append(clean_lat)
    noisy_z.append(noisy_lat)

noisy_z = np.array(noisy_z)
clean_z = np.array(clean_z)
for i in range(0,65):
    noisy_z[i] = noisy_z[i][:850,:]
    clean_z[i] = clean_z[i][:850,:]

"""



