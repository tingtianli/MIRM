
import torch
import torch.utils.data as Data
from torchsample import transforms as tensor_tf
import glob
from skimage import io
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy
import skimage.feature
import os
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy import interpolate
from torchsample import transforms as tensor_tf


            
            


        
def data_parepare(folder_dir):
    
#    data_transform = tensor_tf.Compose([
#          tensor_tf.RandomCrop((128,128)),
#
#      ])
#    

    w,h=512,512
#    w,h=256,256
    
#    w_used,h_used=512,512
    w_used,h_used=256,256
    LF=np.zeros((5,w,h))

#    LF=np.zeros((5,h,w))
    name=folder_dir+'/3_3.png'
    img=io.imread(name,as_gray=True)
    img=cv2.resize(img, (256,256))    
    img=cv2.resize(img, (w,h)) 
    LF[0,:,:]=img
    
    img_color=io.imread(name)/255. 
    img_color=cv2.resize(img_color,(w_used,h_used), interpolation = cv2.INTER_CUBIC)
    img_color[img_color>1]=1
    img_color[img_color<0]=0
        
    name=folder_dir+'/2_2.png'
    img=io.imread(name,as_gray=True)
    img=cv2.resize(img, (256,256))    
    img=cv2.resize(img, (w,h)) 
    LF[1,:,:]=img

    name=folder_dir+'/2_4.png'
    img=io.imread(name,as_gray=True)
    img=cv2.resize(img, (256,256))    
    img=cv2.resize(img, (w,h)) 
    LF[2,:,:]=img

        
    name=folder_dir+'/4_2.png'
    img=io.imread(name,as_gray=True)
    img=cv2.resize(img, (256,256))    
    img=cv2.resize(img, (w,h)) 
    LF[3,:,:]=img
        
    name=folder_dir+'/4_4.png'
    img=io.imread(name,as_gray=True)
    img=cv2.resize(img, (256,256))    
    img=cv2.resize(img, (w,h)) 
    LF[4,:,:]=img
            
        

    gray=(LF[0,:,:])
    gray=cv2.resize(gray, (w_used,h_used)) 
    sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=5)
    mag=np.sqrt(sobelx**2+sobely**2)


    img_color=img_color.transpose(2,0,1)
    img_color=torch.FloatTensor(img_color) 

    LF=torch.FloatTensor(LF)   
    gray=torch.FloatTensor(gray)  
    mag=torch.FloatTensor(mag)  

    
    gray=gray.unsqueeze(0)
    mag=mag.unsqueeze(0)
    


    return img_color,LF,gray,mag





class CustomDataset(Data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self,folder_list):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.folder_list = folder_list



    def __len__(self):
        le=len(self.folder_list)
        return le
    
    def __getitem__(self, idx):
#        idx1=idx

        folder = self.folder_list[idx] +'/syn'
        img_color,LF,gray,mag = data_parepare(folder)
         
        return img_color,LF,gray,mag
    
    
    
    
    
    
    
    
    
    
    
    
    
    