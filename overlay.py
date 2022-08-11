import cv2
import numpy as np
import os
import sys
print(sys.version)

VID = 'fig_cst1'
def main():
    horizontal = []

    IMG_Path = "/home/yasser/Desktop/DATA360/imagesss/"
    SAL = '/home/yasser/Desktop/PhD/360 project/saliency/saliency/'
    #SAL = "/home/yasser/Desktop/TEST_PAPER/saliency/expert/044/"
    image_list = os.listdir(IMG_Path)
    image_list.sort()



    for img in image_list:
        image_path = IMG_Path + img
        sal_path = SAL +img[:-4]+'.png'
        print(image_path)
        print(sal_path)
        inpt = cv2.imread(image_path)

        #inpt = cv2.resize(inpt, (640, 320))
        sa = cv2.imread(sal_path)
    

        sa = cv2.resize(sa,(inpt.shape[1], inpt.shape[0]))
        print(sa.shape)
        print(inpt.shape)

        one = ProduceOverlayed(inpt, sa, "fig_atsal", img)


def ProduceOverlayed(X, prediction, model_name, i):

    if not os.path.exists("./{}".format(model_name)):
        os.makedirs("./{}".format(model_name))

    Y = cv2.applyColorMap(prediction, 11)
    #X = cv2.resize(X, (Y.shape[1], Y.shape[0]), interpolation=cv2.INTER_AREA)
    fin = cv2.addWeighted(Y, 0.299, X, 0.587, 0.114)

    cv2.imwrite("./{}/{}".format(model_name, i), fin)

    return(fin)


main()