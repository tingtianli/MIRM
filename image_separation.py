#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 15:13:48 2017

@author: li
"""

from __future__ import print_function
import sys

sys.path.insert(0,'./models')
sys.path.insert(0,'./custom')

import argparse
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.autograd import Variable
import cv2
import custom_data_simulation_5views as cd
import matplotlib.pyplot as plt
from pure_conv import Disparity_Net
import numpy as np
from skimage import io
from sklearn.cluster import KMeans
from unet_full_cat3 import UNet as _netG3
import glob

model_dir='./model_para/'
parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--nc', type=int, default=6)
parser.add_argument('--outch', type=int, default=3)
parser.add_argument('--niter', type=int, default=100000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--nBottleneck', type=int,default=4000,help='of dim for bottleneck of encoder')
parser.add_argument('--overlapPred',type=int,default=4,help='overlapping edges')
parser.add_argument('--nef',type=int,default=64,help='of encoder filters in first conv layer')
parser.add_argument('--wtl2',type=float,default=0.995,help='0 means do not use else use with this weight')
parser.add_argument('--wtlD',type=float,default=0.001,help='0 means do not use else use with this weight')
parser.add_argument('--input_ch', type=int,default=5)
parser.add_argument('--output_ch', type=int,default=1)
parser.add_argument('--Disparity_Model_pkl',default=model_dir+'Model_Disparity_5views_four_closest_corners_mag.pkl')
parser.add_argument('--netG_2sperated_B_hole_pkl',default=model_dir+'net_img_rec.pkl')
# The network parameters are fine-tuned. The PSNR of results is higher than the paper
parser.add_argument('--ne_rec_B_edge_pkl',default=model_dir+'netG_of_WGAN.pkl')  
parser.add_argument('--test_imgs_folder',default= './imgs_test/')  

opt = parser.parse_args()



cudnn.benchmark = True
opt.cuda=True

def tensor_img_show_c(tensor,num=0):
    result=(tensor.data).cpu().numpy()
    result=result[num,:,:,:]
    result1=result.transpose(1,2,0)
    result=(result1-np.min(result1))/(np.max(result1)-np.min(result1))
    plt.figure()
    plt.imshow(result)
    
    return result1


    
def DMAP_generation_BR(disparity,img,cof):
    MAP=disparity+0
    MAP_B=MAP+0
    MAP_R=MAP+0
    disparity=(disparity.data).cpu().numpy()
    img=(img.data).cpu().numpy()
    
    for i in range(0,disparity.shape[0]):
        ref=disparity[i,0,:,:]
        ind=np.where(ref!=0)
        km= KMeans(n_clusters=3, random_state=0).fit((ref[ind]).reshape(-1, 1) )
 
        TH1=np.amax(km.cluster_centers_)
        TH2=np.amin(km.cluster_centers_)

        TH_1=TH1-(TH1-TH2)*(0.2)
        TH_2=TH2+(TH1-TH2)*(0.2)
    
    return TH_1,TH_2



def imge_mean(B_ini,original):
    B_ini2= B_ini+0
    for k in range(0,3):
        B_ini2[:,:,k]=B_ini[:,:,k]+np.average(original[:,:,k])-np.average(B_ini[:,:,k])

    return B_ini2  

# load networks
# load disparity network
Disparity_Model = Disparity_Net(opt)
Disparity_Model.load_state_dict(torch.load(opt.Disparity_Model_pkl))
Disparity_Model.cuda()

opt.nc=6
opt.outch=3        

# load image reconstruction network
netG_B = _netG3(opt)
netG_B.load_state_dict(torch.load(opt.netG_2sperated_B_hole_pkl))
netG_B.cuda()

# load edge regeneration network
opt.nc=9
netG_rec_edge = _netG3(opt)
netG_rec_edge.load_state_dict(torch.load(opt.ne_rec_B_edge_pkl))
netG_rec_edge.cuda()



Disparity_Model.eval()
netG_B.eval()
netG_rec_edge.eval()



batch1=range(1,11)
batch2=range(1,11)
folder_list =  sorted(glob.glob(opt.test_imgs_folder+'*'))
img_set=cd.CustomDataset(folder_list)

for idx in range(0, 10):
# idx=8  #  select the image pair from 0 to 9
    img_color,LF,gray,mag = img_set[idx]
    
    
    img_color=img_color.unsqueeze(0)
    LF=LF.unsqueeze(0)
    gray=gray.unsqueeze(0)
    mag=mag.unsqueeze(0)
    
    
    img_color=Variable(img_color).cuda()
    LF=Variable(LF).cuda()
    gray=Variable(gray).cuda()
    mag=Variable(mag).cuda()
    
    num=0
    disparity =Disparity_Model(LF)             # disparity estimation
    disparity=(disparity.data).cpu().numpy()  
    disparity=(disparity[0,0,:,:])
    disparity=cv2.resize(disparity, (256, 256))
    disparity=torch.FloatTensor(disparity) 
    disparity=(disparity.unsqueeze(0)).unsqueeze(0)
    disparity=Variable(disparity).cuda()
    
    MMAP=(mag>3).float()   
    cat_MMAP=torch.cat((MMAP,MMAP,MMAP),dim=1)
    disparity=disparity*MMAP
    TH_1,TH_2=(DMAP_generation_BR(disparity,disparity,0.2))  # thresholds generation
            
    DMAP_B=disparity>TH_1
    DMAP_R=disparity<TH_2
    
    MAP1_F=DMAP_B.float()*MMAP
    MAP2_F=DMAP_R.float()*MMAP
    MAP1_F_c=torch.cat((MAP1_F,MAP1_F,MAP1_F),dim=1)
    ref_M_B=img_color*MAP1_F
    ref_M_R=img_color*MAP2_F
    ref_edge=img_color*cat_MMAP
    Input_rec_B_edge=torch.cat((ref_edge,ref_M_B,ref_M_R),dim=1)  # edge regeneration
    rec_B_edge=netG_rec_edge(Input_rec_B_edge)
    rec_R_edge=ref_edge-rec_B_edge
    mask_B2=(rec_B_edge>0.05).float()*cat_MMAP
    mask_B2=((mask_B2[0,0,:,:]+mask_B2[0,1,:,:]+mask_B2[0,2,:,:])>2).float()
    mask_B2=mask_B2.unsqueeze(0).unsqueeze(0)
    mask_B4=torch.cat((mask_B2,mask_B2,mask_B2),dim=1)
    mask_R4=cat_MMAP-mask_B4    
    
    rec_B_edge2=rec_B_edge*cat_MMAP
    rec_R_edge2=rec_R_edge*cat_MMAP
    ref_M_B2=img_color*mask_B4
    ref_M_R2=img_color*mask_R4
    
    Input_B=torch.cat((img_color-ref_M_R2,ref_M_B2),dim=1)
    
    
    fake_B= netG_B(Input_B)  # background reconstruction
    fake_R=img_color-fake_B
    tensor_img_show_c(fake_B)
    tensor_img_show_c(fake_R)
    fake_B=(fake_B.data).cpu().numpy()
    fake_B=fake_B[num,:,:,:]
    fake_B=fake_B.transpose(1,2,0)
    fake_B_CPU=fake_B
        
    fake_R=(fake_R.data).cpu().numpy()
    fake_R=fake_R[num,:,:,:]
    fake_R=fake_R.transpose(1,2,0)
    fake_R_CPU=fake_R
        
    if idx==2:
       a=fake_R_CPU
       fake_R_CPU=fake_B_CPU
       fake_B_CPU=a
            
    gt1 = img=io.imread(folder_list[idx] + '/gt1.png')
    gt2 = img=io.imread(folder_list[idx] + '/gt2.png')
    fake_B_CPU=imge_mean(fake_B_CPU, gt1)  #change the results' mean value are normalized to the ground truth.
    fake_R_CPU=imge_mean(fake_R_CPU, gt2)

# The image downsampling stategies of python and MATLAB are different.  
# Please use MATLAB to calculate PSNR.
