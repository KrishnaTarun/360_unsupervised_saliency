import torch
import os
import numpy as np

from torch.utils import data
from torchvision import utils, transforms
from PIL import Image
from augment import AugmentImage
import glob
import math
import matplotlib.pyplot as plt

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
        sample1, sample2 = self.transform(sample)
        

       return sample1, sample2, frame_index

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

if __name__=="__main__":

    

    train_set = ImageData(
                pil_loader, 
                './check1/',
                transform=AugmentImage())
    # Tf = Projection()

    train_loader = data.DataLoader(train_set, batch_size=1,
                                    num_workers=4,
                                    drop_last=True)
    import os
    # if not os.path.isdir("check1"):
    #     os.makedirs("check1")
    mean=[0.485, 0.456, 0.406]
    std= [0.229, 0.224, 0.225]
    for i, (x1, x2, index) in enumerate(train_loader):
        print(x1.shape, x2.shape)
        img1 = x1[0]
        img2 = x2[0]

        img1 = img1 * torch.tensor(std).view(3, 1, 1)
        img1 = img1 + torch.tensor(mean).view(3, 1, 1)
        img1 = transforms.ToPILImage(mode='RGB')(img1)
        img1 = img1.convert('RGB')
        img1.save(str(i), "JPEG")
        # img1.show()

        img2 = img2 * torch.tensor(std).view(3, 1, 1)
        img2 = img2 + torch.tensor(mean).view(3, 1, 1)
        img2 = transforms.ToPILImage(mode='RGB')(img2)
        img2 = img2.convert('RGB')
        img2.save(str(i)+'_t', "JPEG")
        # img2.show()



