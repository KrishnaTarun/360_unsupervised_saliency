import torch
import os
import datetime
import numpy as np
import cv2
from torch.utils import data
from torchvision import utils
from skimage.transform import resize
from PIL import Image
from model import SalGAN
# The DataLoader for our specific datataset with extracted frames
class Static_dataset(data.Dataset):

    def __init__(self, split, root_path, load_gt=True, resolution=None, val_perc=0.2):
          # augmented frames
        print(root_path)
        self.frames_path = os.path.join(root_path, 'images')
        self.load_gt = load_gt

        if load_gt:
            # ground truth
            self.gt_path = os.path.join(root_path, "saliency")
            self.fx_pah = os.path.join(root_path, "fixation")

        self.resolution = resolution
        self.frames_list = []
        self.gt_list = []
        self.fx_list = []

        # Gives accurate human readable time, rounded down not to include too many decimals
        print('start load data')
        self.frames_list = os.listdir(self.frames_path)
        # frame_files = os.listdir(os.path.join(self.frames_path))
        # self.frames_list = sorted(frame_files, key=lambda x: int(x.split(".")[0]))
        # self.frames_list = self.frames_list[:number_of_frames]
        print(' load images data')
        if load_gt:
            self.gt_list = os.listdir(self.gt_path)
            self.fx_list = os.listdir(self.fx_pah)
            # self.gt_list = sorted(gt_files, key=lambda x: int(x.split(".")[0]))
            # self.gt_list = self.gt_list[:number_of_frames]
            if(len(self.frames_list)!=len(self.gt_list)):
                raise Exception('image list and saliency images are of different sizes' )
            print(' load groundtruth data')
            
        print('data loaded')


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.frames_list)
        # return len(self.frames_list)

    def __getitem__(self, frame_index):

        'Generates one sample of data'

        frame = self.frames_list[frame_index]
        


        if self.load_gt:
            gt = self.gt_list[frame_index]

            fx = self.fx_list[frame_index]

        path_to_frame = os.path.join(self.frames_path, frame)
        
        X = cv2.imread(path_to_frame)

        #X = np.load(path_to_frame)
        #X = cv2.resize(X, self.resolution)

        X = X.astype(np.float32)    
        # X = X - [0.485, 0.456, 0.406]
        X = torch.cuda.FloatTensor(X)
        X = X.permute(2, 0, 1)

        # add batch dim
        # data = X.unsqueeze(0)
        # torchvision.utils.save_image(torchvision.utils.make_grid(data, nrow=1), fp='f.jpg') 
        # data = X
        # Load and preprocess ground truth (saliency maps)
        if self.load_gt:

            path_to_gt = os.path.join(self.gt_path, gt)
            path_to_fx = os.path.join(self.fx_pah, fx)

            # Load as grayscale

            #y = np.load(path_to_gt)
            y = cv2.imread(path_to_gt,0)
            # y = np.load(path_to_gt)
            
            #y = cv2.resize(y, self.resolution)

            y = y/255.0
            
            y = y.astype(np.float32)
            
            y = (y - np.min(y)) / (np.max(y) - np.min(y))
            
            y = torch.cuda.FloatTensor(y)
            gt = y.unsqueeze(0)

            f = cv2.imread(path_to_fx,0)
            # y = np.load(path_to_gt)
            
            #y = cv2.resize(y, self.resolution)

            f = f/255.0
            
            f = f.astype(np.float32)
            
            #f = (f - np.min(f)) / (np.max(f) - np.min(f))
            
            f = torch.cuda.FloatTensor(f)
            gf = f.unsqueeze(0)

          
        # if self.load_gt:
        #     y = y.astype(np.float32)
        #     #y = y/255.0
        #     y = (y - np.min(y)) / (np.max(y) - np.min(y))
        #     y = torch.cuda.FloatTensor(y)
        #     # y = y.permute(2,0,1)
        #     gt = y.unsqueeze(0)

       

            packed = (X, gt, gf)  # pack data with the corresponding  ground truths
        else:
            packed = (X, "_")

        return packed


#         for file in os.listdir('.'):
#       2     a = np.load(file)
# ----> 3     from skimage.transform import rescale, resize
#       4     a =resize(a, (160, 320))
#       5     np.save(file, a)