from functools import partial
import numpy as np
from numpy import random
import time
from skimage import exposure
from skimage.transform import resize
import cv2
import re, os, glob
import cv2
import torch
import scipy.misc
from torchvision import utils

EPSILON = np.finfo('float').eps



from skimage.transform import resize


if __name__ == "__main__":
    from time import time

    

    PM_PATH = "./results_val/"
    image_list = os.listdir(PM_PATH)
    print(image_list)
    i = 0
    for img in image_list:
        try:
            predicted = PM_PATH +img
            predicted = np.load(predicted)
            predicted = (predicted)/predicted.max()
            predicted = cv2.resize(predicted,(320, 160))
            cv2.imshow('kk', predicted)
            cv2.waitKey(0)
        except:
            continue