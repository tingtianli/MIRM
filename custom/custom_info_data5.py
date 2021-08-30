
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
        result=result.squeeze(0)
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
        result=result.squeeze(0)
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

#        disaprity6=np.zeros([height,width])
#    plt.figure()
#    plt.imshow(r_yu)          
#    plt.figure()
#    plt.imshow(r_xv)     
#    plt.figure()
#    plt.imshow(disaprity2,cmap='gray')    
#    plt.figure()
#    plt.imshow(disaprity3,cmap='gray')    
#    plt.figure()
#    plt.imshow(disaprity4,cmap='gray')    
#    plt.figure()
#    plt.imshow(disaprity5,cmap='gray')    
#    plt.figure()
    r=np.stack((r_yu,r_xv),axis=0)
    r_cob=r_yu*(r_yu>r_xv)+r_xv*(r_xv>r_yu)
    r_cob=np.expand_dims(r_cob,axis=0)
    return r,r_cob

def img_shifts(imgs,shift_pixel,selected,ref_ind):

    length=int(np.sqrt(imgs.shape[0]))
    h=imgs.shape[1]
    w=imgs.shape[2]
    x = np.arange(0,h)
    y = np.arange(0,w)
#    xx, yy = np.meshgrid(x, y)
    img_refocused =0
    img_shift=np.zeros(imgs.shape)
    for u in range(0,length):
        for v in range(0,length):
            f = interpolate.interp2d(x, y, imgs[u*length+v,:,:], kind='cubic')
            xnew=x+(selected[v]-selected[ref_ind])*shift_pixel
            ynew=y+(selected[u]-selected[ref_ind])*shift_pixel
            img_shift[u*length+v,:,:]=f(xnew, ynew)
            img_refocused =img_refocused + img_shift[u*length+v,:,:]/(length*length)
#            
#    plt.figure()
#    plt.imshow(img_refocused)         
    return img_shift
            
            
            


        
def data_parepare(folder_dir1,folder_dir2,data_transform1=None,data_transform2=None,
                  affine_transform1=None,affine_transform2=None):
    

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
#       try:
        LF1=data_transform1(LF1)
#       except:
#           LF1=LF1
               
       
    if data_transform2 is not None: 
#        try:
        LF2=data_transform2(LF2)
#        except:
#           LF2=LF2  
           
#    coef2=random.random()*0.2+0.3
#    coef2=0.4
#    coef1=1-coef2
    
    
    coef2=1
    coef1=1
    
#    img1_color=LF1[0:3,:,:]*coef1
#    img2_color=LF2[0:3,:,:]*coef2      
#    disparity1=LF1[3,:,:]
#    disparity2=LF2[3,:,:] 
#    mag1=LF1[4,:,:]*coef1
#    mag2=LF2[4,:,:]*coef2 
#    gray1=LF1[5,:,:]*coef1
#    gray2=LF2[5,:,:]*coef2 
    
    disparity1=(LF1[3,:,:]).numpy()   
    disparity2=(LF2[3,:,:]).numpy()  
    if np.mean(disparity1)<np.mean(disparity2):
        LF1,LF2=LF2,LF1
    
    img1_color=(LF1[0:3,:,:]).numpy()*coef1
    img2_color=(LF2[0:3,:,:]).numpy()*coef2 
#    img1_color=img1_color*coef1
#    img2_color=img2_color*coef2
    
    disparity1=(LF1[3,:,:]).numpy()   
    disparity2=(LF2[3,:,:]).numpy()  
    disparity2=disparity2-(np.mean(disparity1)-np.mean(disparity2))*0.5
#    disparity2=disparity2-(np.mean(disparity1)-np.mean(disparity2))*0
#    min_d=np.min(disparity2)

    
#    d1,d2=np.max(disparity1),np.min(disparity1)
#    d3,d4=np.max(disparity2),np.min(disparity2)z
#    if d3-d1<d4-d2:
#        disparity2=disparity2-d3+d1-(np.mean(disparity1)-np.mean(disparity2))*0.8
#    else:
#        disparity2=disparity2-d4+d2-(np.mean(disparity1)-np.mean(disparity2))*0.8
            

#    mag1=(LF1[4,:,:]*coef1).numpy()   
#    mag2=(LF2[4,:,:]*coef2 ).numpy()
    
#    gray1=(LF1[4,:,:]*coef1).numpy()   
#    gray2=(LF2[4,:,:]*coef2 ).numpy()
    
    gray1=(LF1[4,:,:]*1).numpy()*coef1
    gray2=(LF2[4,:,:]*1 ).numpy()*coef2
    
    
    w,h=256,256
    
    img1_color=img1_color.transpose(1,2,0)
    img2_color=img2_color.transpose(1,2,0)
    img1_color=cv2.resize(img1_color, (w,h)) 
    img2_color=cv2.resize(img2_color, (w,h)) 
    
    sobelx = cv2.Sobel(gray1+gray2,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(gray1+gray2,cv2.CV_64F,0,1,ksize=5)
    ref_mag=np.sqrt(sobelx**2+sobely**2)
    ref_mag=torch.FloatTensor(cv2.resize(ref_mag, (w,h))) 

    
    sobelx = cv2.Sobel(gray1,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(gray1,cv2.CV_64F,0,1,ksize=5)
    ref1_mag=np.sqrt(sobelx**2+sobely**2)
    ref1_mag=torch.FloatTensor(cv2.resize(ref1_mag, (w,h))) 
#    ref1_mag=ref1_mag.unsqueeze(0)
    
    sobelx = cv2.Sobel(gray2,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(gray2,cv2.CV_64F,0,1,ksize=5)
    ref2_mag=np.sqrt(sobelx**2+sobely**2)
    ref2_mag=torch.FloatTensor(cv2.resize(ref2_mag, (w,h))) 
#    ref2_mag=ref2_mag.unsqueeze(0)
    
    
    img1_color=torch.FloatTensor(img1_color.transpose(2,0,1))
    img2_color=torch.FloatTensor(img2_color.transpose(2,0,1))
    disparity1=torch.FloatTensor(cv2.resize(disparity1, (w,h))) 
    disparity2=torch.FloatTensor(cv2.resize(disparity2, (w,h))) 
#    mag1=torch.FloatTensor(cv2.resize(mag1, (w,h)))
#    mag2=torch.FloatTensor(cv2.resize(mag2, (w,h)))
    
    
    gray1=torch.FloatTensor(cv2.resize(gray1, (w,h))) 
    gray2=torch.FloatTensor(cv2.resize(gray2, (w,h))) 
    
    
    gray=gray1+gray2
#    
#    disparity1=disparity1*(ref1_mag>3).float()
#    disparity2=disparity2*(ref2_mag>3).float()
#    
#    ind1=np.nonzero(disparity1.numpy())
#    ind2=np.nonzero(disparity2.numpy())
#    
#    if (np.sum(ind1)!=0) and (np.sum(ind2)!=0):
#        
#        minval1 = np.min(disparity1.numpy()[ind1])
#        maxval1 = np.max(disparity1.numpy()[ind1])
#    
#        minval2 = np.min(disparity2.numpy()[ind2])
#        maxval2 = np.max(disparity2.numpy()[ind2])
#    
#        rg=np.max([maxval1,maxval2])-np.min([minval1,minval2])
#        val=0
#        while True:
#            remin2=minval2+val
#            remax2=maxval2+val
#            if remax2<maxval1 and remin2<minval1:
#                break
#            val=(random.random()*0.5-1)*2*rg
#        disparity2=disparity2+val
    
    th=3
#    th=0
    disparity1=disparity1*(ref1_mag>th).float()*(ref1_mag>=ref2_mag).float()
    disparity2=disparity2*(ref2_mag>th).float()*(ref1_mag<ref2_mag).float()
    
    disparity=disparity1*(ref1_mag>th).float()*(ref1_mag>=ref2_mag).float()+disparity2*(ref2_mag>th).float()*(ref1_mag<ref2_mag).float()
    
    ref=img1_color+img2_color
    
    ref_mag=ref_mag.unsqueeze(0)
    ref1_mag=ref1_mag.unsqueeze(0)
    ref2_mag=ref2_mag.unsqueeze(0)
#    ref_dx=gradx(ref)
#    ref_dy=grady(ref)
#    
#    ref_dx=torch.FloatTensor(ref_dx)
#    ref_dy=torch.FloatTensor(ref_dy)
    
    disparity=disparity.unsqueeze(0)
    disparity1=disparity1.unsqueeze(0)
    disparity2=disparity2.unsqueeze(0)
    
#    mag2=mag2.unsqueeze(0)
    gray=gray.unsqueeze(0)
#    gray2=gray2.unsqueeze(0)
           
    return ref,ref1_mag,ref2_mag,ref_mag,disparity,disparity1,disparity2,img1_color,img2_color,gray



class CustomDataset(Data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self,folder_list,folder_list2,
                 data_transform1=None,data_transform2=None,affine_transform1=None,affine_transform2=None,gray=False,idx2=None,extreflection=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.folder_list= folder_list
        self.folder_list2= folder_list2
        self.data_transform1 = data_transform1
        self.data_transform2 = data_transform2
        self.affine_transform1 = affine_transform1
        self.affine_transform2 = affine_transform2
        self.idx2=idx2
        self.extreflection=extreflection


    def __len__(self):
        le=len(self.folder_list)
        return le
    
    def __getitem__(self, idx):
        idx1=idx
        if self.idx2 is None:
            idx2_r=random.randrange(0,len(self.folder_list2))
        else:
            idx2_r=self.idx2
        while True:
            ref,ref1_mag,ref2_mag,ref_mag,disparity,disparity1,disparity2,img1_color,img2_color,gray = data_parepare(self.folder_list[idx1],self.folder_list2[idx2_r],\
                                                data_transform1=self.data_transform1,data_transform2=self.data_transform2)
            if disparity1.sum()!=0 and disparity2.sum()!=0:
                break;
        
        
         
        return ref,ref1_mag,ref2_mag,ref_mag,disparity,disparity1,disparity2,img1_color,img2_color,gray
    