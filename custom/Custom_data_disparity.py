
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


def gradx(input,cpu=True):
#    input=torch.FloatTensor(input)
#    input=input.unsqueeze(0)
    input=input.unsqueeze(0)
    input=Variable(input)
    filter_dx=torch.autograd.Variable(torch.zeros(input.size()[1],input.size()[1],1,3))
    dx=torch.from_numpy(np.array([0,-1,1],np.float32))
    for k1 in range(0,input.size()[1]):
        for k2 in range(0,input.size()[1]):
            filter_dx[k1,k2,0,:]=torch.autograd.Variable(dx)     
    result=F.conv2d(input,filter_dx,padding=(0,1))
    if cpu is True:
        result=(result.data.cpu()).numpy()
    return result
    
def grady(input,cpu=True):
#    input=torch.FloatTensor(input)
#    input=input.unsqueeze(0)
    input=input.unsqueeze(0)
    input=Variable(input)
    filter_dy=torch.autograd.Variable(torch.zeros(input.size()[1],input.size()[1],3,1))
    dy=torch.from_numpy(np.array([[0],[-1],[1]],np.float32))
    for k1 in range(0,input.size()[1]):
        for k2 in range(0,input.size()[1]):
            filter_dy[k1,k2,:,:]=torch.autograd.Variable(dy) 
    result=F.conv2d(input,filter_dy,padding=(1,0))
    if cpu is True:
        result=(result.data.cpu()).numpy()
    return result

def EPI_generation_yu(imgs,y,u):
    return imgs[u:u+8,y,:]


def EPI_generation_xv(imgs,x,v):
    u_ind=np.arange(0,8)*8
    return imgs[u_ind+v,:,x]

def sharpen(imgs):
    length=imgs.shape[0]
    sharpened=np.zeros(imgs.shape)
    for i in range(0,length):
        blurred_f = cv2.GaussianBlur(imgs[i,:,:],(3,3),0.5)
#        filter_blurred_f = scipy.ndimage.gaussian_filter(blurred_f, 1)
        alpha = 5
        sharpened[i,:,:] = imgs[i,:,:]+ alpha * (imgs[i,:,:] - blurred_f)
    return sharpened

def Reliability_ST(imgs,fixed_u,fixed_v,h,w):
    e=0.000000001
    height=h
    width=w
    sita=1
    r_yu=np.zeros([height,width])
    r_xv=np.zeros([height,width])
    for y in range(0,height):
        EPI=EPI_generation_yu(imgs,y,fixed_u)
        Jxx, Jxy, Jyy=skimage.feature.structure_tensor(EPI, sigma=sita, mode='nearest')
        for x in range(0,width):
            r_yu[y,x]=((Jyy[fixed_v,x]-Jxx[fixed_v,x])**2+4*Jxy[fixed_v,x]**2)  / (Jxx[fixed_v,x]+Jyy[fixed_v,x]+e)**2
            
    for x in range(0,width):
        EPI=EPI_generation_xv(imgs,x,fixed_v)
        Jxx, Jxy, Jyy=skimage.feature.structure_tensor(EPI, sigma=sita, mode='nearest')
        for y in range(0,height):      
            r_xv[y,x]=((Jyy[fixed_u,y]-Jxx[fixed_u,y])**2+4*Jxy[fixed_u,y]**2)  / (Jxx[fixed_u,y]+Jyy[fixed_u,y]+e)**2


    r=np.stack((r_yu,r_xv),axis=0)
    r_cob=r_yu*(r_yu>r_xv)+r_xv*(r_xv>r_yu)
    r_cob=np.expand_dims(r_cob,axis=0)
    return r,r_cob

def img_shifts(imgs,shift_pixel,selected_u,selected_v):

    h=imgs.shape[1]
    w=imgs.shape[2]
    x = np.arange(0,h)
    y = np.arange(0,w)

    img_shift=np.zeros(imgs.shape)
    for u in range(0,len(selected_u)):
        v=u
        f = interpolate.interp2d(x, y, imgs[u,:,:], kind='cubic')
        xnew=x+(selected_u[v]-selected_u[0])*shift_pixel
        ynew=y+(selected_v[u]-selected_v[0])*shift_pixel
        img_shift[u,:,:]=f(xnew, ynew)
      
    return img_shift
            
            
            


        
def data_parepare(folder_dir1,folder_dir2,data_transform1=None,data_transform2=None,affine_transform1=None,affine_transform2=None,shift=False):
    

    LF1=np.load(folder_dir1)
#    file_dir_img=file_dir2
    LF2=np.load(folder_dir2)
    
            
    LF1=torch.FloatTensor(LF1)
    LF2=torch.FloatTensor(LF2)
    
    if affine_transform1 is not None:
       LF1=affine_transform1(LF1)
       
    if affine_transform2 is not None:
       LF2=affine_transform2(LF2)
        
    if data_transform1 is not None:
       try:
           LF1=data_transform1(LF1)
       except:
           LF1=LF1
               
       
    if data_transform2 is not None: 
        try:
            LF2=data_transform2(LF2)
        except:
           LF2=LF2    

    selected_u=np.array([6,4,2,8,10])
    selected_v=np.array([6,4,2,8,10])
           
#    selected_u=np.array([2,0,0,4,4])
#    selected_v=np.array([2,0,4,0,4])

    if  shift is True:
        shift_pixel=random.random()
        shifted_LF2=img_shifts(LF2.numpy(),-2*shift_pixel,selected_u,selected_v)
        shifted_LF2=torch.FloatTensor(shifted_LF2)
    else:
        shifted_LF2=LF2
        
    coef2=random.random()*0.2+0.3
    coef1=1-coef2 

    LF2=shifted_LF2
    LF=coef1*LF1+coef2*LF2
        


    dx=np.array([0,-1,1])
    dy=np.array([[0],[-1],[1]])
    l=LF.size()[0]

    mag=np.zeros(LF.size())
    for k in range(0,l):
        img=(LF[k,:,:]).numpy()
        gdx=cv2.filter2D(img,-1,dx)
        gdy=cv2.filter2D(img,-1,dy)
        mag_ref=np.sqrt(gdx**2+gdy**2)

#        sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
#        sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
#        mag_ref=np.sqrt(sobelx**2+sobely**2)
        mag[k,:,:]=mag_ref
#        
    mag=torch.FloatTensor(mag)  
#    mag=mag_ref.unsqueeze(0)
    
    return LF,mag,LF1,LF2





class CustomDataset(Data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self,folder_list,
                 data_transform1=None,data_transform2=None,affine_transform1=None,affine_transform2=None,gray=False,idx2=None,shift=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.folder_list= folder_list
        self.data_transform1 = data_transform1
        self.data_transform2 = data_transform2
        self.affine_transform1 = affine_transform1
        self.affine_transform2 = affine_transform2
        self.idx2=idx2
        self.shift=shift


    def __len__(self):
        le=len(self.folder_list)
        return le
    
    def __getitem__(self, idx):
        idx1=idx
        if self.idx2 is None:
            idx2_r=random.randrange(0,len(self.folder_list))
        else:
            idx2_r=self.idx2

        LF,mag,LF1,LF2 = data_parepare(self.folder_list[idx1],self.folder_list[idx2_r],\
                                                data_transform1=self.data_transform1,data_transform2=self.data_transform2,shift=self.shift)
         
        return LF,mag,LF1,LF2
    
