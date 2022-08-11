import cv2
import numpy as np
import os

slid = 4
images = 'path to images'
gt = 'path to gt'
fx = 'path to fixatio'

path='./dataset/training_set/image/'

stored_path = './dataset/augment/image/'

if not os.path.isdir(stored_path):
  os.makedirs(stored_path)

frame_files_sorted = sorted(os.listdir(path), key = lambda x: (x.split(".")[0])[:3] )
frame_files_sorted = [img for img in frame_files_sorted if img.endswith('.jpg')]
print(frame_files_sorted)
j=0
for pa in frame_files_sorted:
  
  if pa.endswith(".jpg"):

    a = (6-len(str(j)))*'0'+str(j)+'.jpg'
    print(a,'save new image')
    img = cv2.imread(path+pa)
    img = cv2.resize(img,(320,160))
    cv2.imwrite(stored_path+a,img)
    fl = cv2.flip(img, 1)
    j+=1
    a = (6-len(str(j)))*'0'+str(j)+'.jpg'

    cv2.imwrite(stored_path+a,fl)
    print(a)
    step =int(img.shape[1]/slid)

  
    high = int(img.shape[0]*0.04)

    upper =img[:high,:,:]
    upper = cv2.resize(upper,(int(img.shape[1]),int(img.shape[0]*0.06)))
    midle = img[high:-high,:,:]
    down =img[-high:,:,:]
    down = cv2.resize(down,(int(img.shape[1]),int(img.shape[0]*0.02)))
    im = np.concatenate((upper,midle,down),axis=0)
    im = cv2.resize(im,(int(img.shape[1]),int(img.shape[0])))
    j+=1
    a = (6-len(str(j)))*'0'+str(j)+'.jpg'
    cv2.imwrite(stored_path+a,im)
    print(a)
#flip
    fl = cv2.flip(im, 1)
    j+=1
    a = (6-len(str(j)))*'0'+str(j)+'.jpg'
    cv2.imwrite(stored_path+a,fl)
    print(a)
    upper =img[:high,:,:]
    upper = cv2.resize(upper,(int(img.shape[1]),int(img.shape[0]*0.02)))
    midle = img[high:-high,:,:]
    down =img[-high:,:,:]
    down = cv2.resize(down,(int(img.shape[1]),int(img.shape[0]*0.06)))
    im = np.concatenate((upper,midle,down),axis=0)
    im = cv2.resize(im,(int(img.shape[1]),int(img.shape[0])))
    j+=1
    a = (6-len(str(j)))*'0'+str(j)+'.jpg'
    cv2.imwrite(stored_path+a,im)
    
    print(a)
#flip
    fl = cv2.flip(im, 1)
    j+=1
    a = (6-len(str(j)))*'0'+str(j)+'.jpg'
    cv2.imwrite(stored_path+a,fl)
    print(a)
    
    print(img.shape[1])
    print(img.shape[0])
    for i in range(step,slid*step,step):

      first_part = img[:,:i,:]
      second_part = img[:,i:,:]
      full = np.concatenate((second_part,first_part),axis=1)
      full2 = cv2.flip(full, 1)
      j+=1
      a = (6-len(str(j)))*'0'+str(j)+'.jpg'
      cv2.imwrite(stored_path+a,full)
      print(a," shift: ",i)
      j+=1
      a = (6-len(str(j)))*'0'+str(j)+'.jpg'
      cv2.imwrite(stored_path+a,full2)
      print(a," shiftflip: ",)
    j+=1

