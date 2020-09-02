import torch
import os
import datetime
import numpy as np

from torch.utils import data
from torchvision import utils, transforms
from PIL import Image

import glob
import math
import matplotlib.pyplot as plt
#--------------------Projection--------------------------
# def T1(src):
#     lng = np.pi*(2 * (np.repeat(np.array([np.arange(0,src.shape[1])]), repeats =
#     src.shape[0], axis = 0)) / (src.shape[1]) - 1.0)
#     lat = 0.5*np.pi*(2 * (np.repeat(np.array(np.ones((1,src.shape[1]))), repeats =
#     src.shape[0], axis = 0)*np.arange(0,src.shape[0]).reshape(src.shape[0],1)) /
#     (src.shape[0]) - 1.0)
#     Z_axes = np.cos(lat) * np.cos(lng)
#     Y_axes = np.cos(lat) * np.sin(lng)
#     X_axes = -np.sin(lat)
#     D = np.sqrt(X_axes*X_axes + Y_axes*Y_axes)
#     lat_shifted = np.arctan2(Z_axes, D)
#     lng_shifted = np.arctan2(Y_axes, X_axes)
#     x_shifted = (0.5 * (lng_shifted) / np.pi + 0.5) * (src.shape[1]) -0.5
#     y_shifted = ((lat_shifted) / np.pi + 0.5) * (src.shape[0]) -0.5
#     first_transformation = src[y_shifted.astype(np.intp),x_shifted.astype(np.intp)]
#     return first_transformation
# def T2(src):
#     lng = np.pi*(2 * (np.repeat(np.array([np.arange(0,src.shape[1])]), repeats =
#     src.shape[0], axis = 0)) / (src.shape[1]) - 1.0)
#     lat = 0.5*np.pi*(2 * (np.repeat(np.array(np.ones((1,src.shape[1]))), repeats =
#     src.shape[0], axis = 0)*np.arange(0,src.shape[0]).reshape(src.shape[0],1)) /
#     (src.shape[0]) - 1.0)

#     Z_axes = np.cos(lat) * np.cos(lng+np.pi)
#     Y_axes = -np.cos(lat) * np.sin(lng+np.pi)
#     X_axes = np.sin(lat)
#     D = np.sqrt(X_axes*X_axes + Y_axes*Y_axes)
#     lat_shifted = np.arctan2(Z_axes, D)
#     lng_shifted = np.arctan2(Y_axes, X_axes)
#     x_shifted = (0.5 * (lng_shifted) / np.pi + 0.5) * (src.shape[1]) -0.5
#     y_shifted = ((lat_shifted) / np.pi + 0.5) * (src.shape[0]) -0.5
#     second_transformation = src[y_shifted.astype(np.intp), x_shifted.astype(np.intp)]
#     # Image.fromarray(second_transformation)
#     return second_transformation
    
class Projection():

    def __init__(self, ht=160, wt=320): 
        super(Projection, self).__init__()
        # ht: height of image
        # wt: width  of image 
        self.ht = ht 
        self.wt = wt
        self.T1, self.T2 = self.initCalculate()
        
    def initCalculate(self):
        #calulate longitude and latitude

        #---------longitude---------------
        lng = np.pi*(2 *(np.repeat(np.array([np.arange(0,self.wt)]),
                         repeats = self.ht, axis = 0)) / (self.wt) - 1.0)
        #----------latitude---------------
        lat = 0.5*np.pi*(2 * (np.repeat(np.array(np.ones((1,self.wt))),
                         repeats = self.ht, axis = 0)*np.arange(0,self.ht).reshape(self.ht,1))/(self.ht) - 1.0)

        lat = torch.from_numpy(lat)
        lng = torch.from_numpy(lng)
        #TODO
        # z = torch.arange(1, self.wt+1).reshape(1, self.wt)
        # z = 2*(torch.repeat_interleave(z, repeats=self.ht, dim=0))/(self.wt)
        # lng = math.pi*(z - 1.0)

        # print(lng.max(), lng.min())
        # y = torch.ones(1, self.wt)
        # y = 2*(torch.repeat_interleave(y, repeats = self.ht, dim=0))
        # y = y*torch.arange(0, self.ht).reshape(self.ht, 1)/self.ht
        # lat = 0.5*math.pi*(y -1.0)
        # print(lat.max(), lat.min())
        
        #-----------calculate Axes T1----------------------
        Z1_axes = torch.cos(lat) * torch.cos(lng)    
        Y1_axes = torch.cos(lat) * torch.sin(lng)
        X1_axes = -torch.sin(lat)

        T1 = self.calShift(Z1_axes, X1_axes, Y1_axes) 

        #----------calculate Axes T2------------------------
        Z2_axes = torch.cos(lat) * torch.cos(lng+math.pi)    
        Y2_axes = -torch.cos(lat) * torch.sin(lng+math.pi)
        X2_axes = torch.sin(lat)
        
        T2 =  self.calShift(Z2_axes, X2_axes, Y2_axes) 

        return T1, T2
    
    def calShift(self, Z, X, Y):

        D = torch.sqrt(X*X + Y*Y)
        lat_shifted = torch.atan2(Z, D)
        lng_shifted = torch.atan2(Y, X)

        x_shifted = (0.5 * (lng_shifted) / math.pi + 0.5) * (self.wt)-0.5  
        y_shifted = ((lat_shifted) / math.pi + 0.5) * (self.ht)-0.5 

        return (x_shifted, y_shifted)

    def __call__(self, x):

        tf1 = x[:, :, self.T1[1].long(), self.T1[0].long()]

        tf2 = x[:, :, self.T2[1].long(), self.T2[0].long()] 

        return tf1, tf2


#--------------------image-----------------------------
class ImageData(data.Dataset):
    def __init__(self,  loader, root_path=None, transform=None):
        super(ImageData,self).__init__()

        self.root_path = root_path
        self.loader = loader
        self.transform = transform

        #list with complete path
        self.samples = glob.glob(os.path.join(root_path,"*.jpg"))
        # print(self.samples)


    def __len__(self):

        #Total images
        return len(self.samples)
    
    def __getitem__(self, frame_index):
        
       frame = self.samples[frame_index]
       
       sample = self.loader(frame)
       
       if self.transform is not None:
        sample = self.transform(sample)
        

       return sample, frame_index

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

if __name__=="__main__":

    train_set = ImageData(
                pil_loader, 
                './data/Train/indoors',
                transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor()
                ])
            )
    Tf = Projection()

    train_loader = data.DataLoader(train_set, batch_size=1,
                                    num_workers=4,
                                    drop_last=True)
    import os
    if not os.path.isdir("check1"):
        os.makedirs("check1")
    for i, (x, index) in enumerate(train_loader):

        tf1, tf2 = Tf(x)
        print(x.shape)
        plt.imsave(os.path.join('check1',str(i)+"_t2"+".jpg"), tf2[0].permute(1, 2, 0).numpy())
        plt.imsave(os.path.join('check1',str(i)+"_t1"+".jpg"), tf1[0].permute(1, 2, 0).numpy())
        plt.imsave(os.path.join('check1',str(i)+".jpg"),x[0].permute(1, 2, 0).numpy())


        #------------numpy testing-----------------------
        # x = x[0].permute(1, 2, 0)
        # x = x.numpy() 
        # tf2 = T2(x)
        # tf1 = T1(x)
        # print(max(tf1.flatten()), min(tf1.flatten()))
        # plt.imsave(os.path.join('check',str(i)+"_t2"+".jpg"), tf2)
        # plt.imsave(os.path.join('check',str(i)+"_t1"+".jpg"), tf1)
        # plt.imsave(os.path.join('check',str(i)+".jpg"), x)
        #-----------------------------------
        
        #-----------------------------------------------------
        #-----------------------------------------------------
        # plt.show()



