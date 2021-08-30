    
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 15:12:29 2018

@author: li
"""
from skimage import io
import numpy as np
import glob
from tqdm import tqdm
from multiprocessing.dummy import Pool
import os.path
import cv2

def list_images(folder, pattern='*'):
    filenames = sorted(glob.glob(folder + pattern))
    return filenames

try:
    os.mkdir('./npy_all_four_closest_corners/')
except:
    print ("folder exists")
    
folder1='./scenes_train/'
folder_dir = list_images(folder1)

for i in tqdm(range(0,len(folder_dir))):
        w,h=625,434 

        height,width=h,w


        leng=5

        LF1=np.zeros((5,height,width))
        name=folder_dir[i]+'/3_3.png'
        img=io.imread(name,as_gray=True)
        img=cv2.resize(img, (w,h)) 
        LF1[0,:,:]=img
        
        name=folder_dir[i]+'/2_2.png'
        img=io.imread(name,as_gray=True)
        img=cv2.resize(img, (w,h)) 
        LF1[1,:,:]=img

        name=folder_dir[i]+'/2_4.png'
        img=io.imread(name,as_gray=True)
        img=cv2.resize(img, (w,h)) 
        LF1[2,:,:]=img
        
        name=folder_dir[i]+'/4_2.png'
        img=io.imread(name,as_gray=True)
        img=cv2.resize(img, (w,h)) 
        LF1[3,:,:]=img
        
        name=folder_dir[i]+'/4_4.png'
        img=io.imread(name,as_gray=True)
        img=cv2.resize(img, (w,h)) 
        LF1[4,:,:]=img
        
        
        patch_size=128
        interval=16
        ind_x=np.arange(0,width-patch_size-interval,interval)
        pool=Pool(2)
        def save_array(ind):
            for idy in np.arange(0,height-patch_size-interval,interval):
                idx=ind_x[ind]
                sub_imgs=LF1[:,idy:idy+patch_size,idx:idx+patch_size]

                file_name='./npy_all_four_closest_corners/'+str(i)+'_'+str(idx)+'_'+str(idy)+\
                '_patch_size_'+str(patch_size)+'_sub_imgs'
                np.save(file_name, sub_imgs)
                
        ind=range(0,len(ind_x))
        pool.map(save_array,ind)       
    
    