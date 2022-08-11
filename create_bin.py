import argparse
import numpy as np
import torch
import os
import cv2
from torch import nn
from torchsummary import summary
from skimage.transform import resize

"""""
def get_parser() :
    parser = argparse.ArgumentParser(description="image saliency predictions. Use this script to get saliency maps for your image and store its value into bin file")
    parser.add_argument('-dst', dest='dst', default="./bin_files/", help="Add root path to output predictions .")
    parser.add_argument('-src', dest='src', default="./test/image/", help="Add root path to your test dataset.")
    parser.add_argument('-model_path', dest='model_path', default="./weight/model_set02.pt", help="Add path to your weight.")
    return parser
"""

#src = "./V/"
src = "/home/yasser/Downloads/SalGAN360-master/scripts/folders/saliency/"
dst = "./bin/"


if not os.path.isdir('./bin'):
    os.makedirs('./bin')

def main() :

    image_list = os.listdir(src)
    image_list.sort()

    for im in image_list:

        #Output = np.load(src+im)
        Output = cv2.imread(src+im,0)
        Output = Output/255.0
        #Output = resize(Output, (1024,2048))
        
        assert len(Output.shape) == 2
        with open(os.path.join(dst,"HEsalmap_"+im[1:3]+"_2048x1024_32b.bin"), 'ab') as the_file:
            for i in range(Output.shape[0]):
                line = np.float32(np.array(Output[i]))
                the_file.write(line)



if __name__ == '__main__':

    main()
"""
import os
import cv2
a = os.listdir('./')
for i in a:
    img = cv2.imread('./'+i)
    img = cv2.resize(img,(2048,1024))
    cv2.imwrite('./'+i,img)
"""
