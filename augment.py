
import os
from PIL import Image
import numpy as np
import torch
import math
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


INTERP = 3
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

        # lat = torch.from_numpy(lat)
        # lng = torch.from_numpy(lng)
        
        #-----------calculate Axes T1----------------------
        Z1_axes = np.cos(lat) * np.cos(lng)    
        Y1_axes = np.cos(lat) * np.sin(lng)
        X1_axes = -np.sin(lat)

        T1 = self.calShift(Z1_axes, X1_axes, Y1_axes) 

        #----------calculate Axes T2------------------------
        Z2_axes = np.cos(lat) * np.cos(lng+math.pi)    
        Y2_axes = -np.cos(lat) * np.sin(lng+math.pi)
        X2_axes = np.sin(lat)
        
        T2 =  self.calShift(Z2_axes, X2_axes, Y2_axes) 

        return T1, T2
    
    def calShift(self, Z, X, Y):

        D = np.sqrt(X*X + Y*Y)
        lat_shifted = np.arctan2(Z, D)
        lng_shifted = np.arctan2(Y, X)

        x_shifted = (0.5 * (lng_shifted) / math.pi + 0.5) * (self.wt)-0.5  
        y_shifted = ((lat_shifted) / math.pi + 0.5) * (self.ht)-0.5 

        return (x_shifted, y_shifted)

    def __call__(self, x, label):

        if(label==0):
            tf1 = np.array(x)[self.T1[1].astype(int), self.T1[0].astype(int)]
            return Image.fromarray(tf1)
        else:
            tf2 = np.array(x)[self.T2[1].astype(int), self.T2[0].astype(int)] 
            return Image.fromarray(tf2)

        # return tf1, tf2


#https://github.com/Philip-Bachman/amdim-public/blob/master/datasets.py
class RandomTranslateWithReflect:
    '''
    Translate image randomly
    Translate vertically and horizontally by n pixels where
    n is integer drawn uniformly independently for each axis
    from [-max_translation, max_translation].
    Fill the uncovered blank area with reflect padding.
    '''

    def __init__(self, max_translation):
        self.max_translation = max_translation

    def __call__(self, old_image):
        xtranslation, ytranslation = np.random.randint(-self.max_translation,
                                                       self.max_translation + 1,
                                                       size=2)
        xpad, ypad = abs(xtranslation), abs(ytranslation)
        xsize, ysize = old_image.size

        flipped_lr = old_image.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_tb = old_image.transpose(Image.FLIP_TOP_BOTTOM)
        flipped_both = old_image.transpose(Image.ROTATE_180)

        new_image = Image.new("RGB", (xsize + 2 * xpad, ysize + 2 * ypad))

        new_image.paste(old_image, (xpad, ypad))

        new_image.paste(flipped_lr, (xpad + xsize - 1, ypad))
        new_image.paste(flipped_lr, (xpad - xsize + 1, ypad))

        new_image.paste(flipped_tb, (xpad, ypad + ysize - 1))
        new_image.paste(flipped_tb, (xpad, ypad - ysize + 1))

        new_image.paste(flipped_both, (xpad - xsize + 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad - xsize + 1, ypad + ysize - 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad + ysize - 1))

        new_image = new_image.crop((xpad - xtranslation,
                                    ypad - ytranslation,
                                    xpad + xsize - xtranslation,
                                    ypad + ysize - ytranslation))
        return new_image

class AugmentImage():
    
    
    def __init__(self):
        
        self.flip_lr = transforms.RandomHorizontalFlip(p=0.5)
        self.project = Projection() 
        self.resize = transforms.Resize((160, 320), interpolation=2)
        # self.crop = transforms.CenterCrop((160, 320))
        self.crop = transforms.RandomResizedCrop((160, 320), scale=(0.75, 0.75),
                                     ratio=(0.7, 1.4), interpolation=INTERP)
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
         
        col_jitter = transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8)

        img_jitter = transforms.RandomApply([
            RandomTranslateWithReflect(10)], p=0.8)
        
        rnd_gray = transforms.RandomGrayscale(p=0.25)
        
        # main transform for self-supervised training
        self.train_transform = transforms.Compose([
            img_jitter,
            col_jitter,
            rnd_gray,
            transforms.ToTensor(),
            normalize
        ])

        # transform for testing (and for not performing augmentation)
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

    def __call__(self, inp):
        inp = self.resize(self.flip_lr(inp))
        out1 = self.test_transform(inp)

        #---Uniformly project----
        label = np.random.randint(2, size=1)
        out2 = self.project(inp, label)
        #------------------------
        
        out2 = self.train_transform(self.crop(out2))
        # out2 = self.test_transform(self.crop(out2))
        # out2 = self.test_transform(out2)
        
        return out1, out2

if __name__=='__main__':

    import os
    mean=[0.485, 0.456, 0.406]
    std= [0.229, 0.224, 0.225]
    augment = AugmentImage()
    pth = 'check_augment'
    if not os.path.isdir(pth):
        os.makedirs(pth)

    for fi in os.listdir('check1'):
        with open('check1/'+fi, 'rb') as f:
            img = Image.open(f)
            img  = img.convert('RGB')

            img1, img2 = augment(img)
            # ---------------------------------------------------
            img1 = img1 * torch.tensor(std).view(3, 1, 1)
            img1 = img1 + torch.tensor(mean).view(3, 1, 1)
            img1 = transforms.ToPILImage(mode='RGB')(img1)
            img1 = img1.convert('RGB')
            # img1.show()
            img1.save(os.path.join(pth, fi))
            #-----------------------------------------------------
            img2 = img2 * torch.tensor(std).view(3, 1, 1)
            img2 = img2 + torch.tensor(mean).view(3, 1, 1)
            img2 = transforms.ToPILImage(mode='RGB')(img2)
            img2 = img2.convert('RGB')
            # img2.show()
            img2.save(os.path.join(pth, fi.split('.')[0]+'_t.jpg'))
        
        #-------------------------------------------------------


