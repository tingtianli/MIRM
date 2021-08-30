#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 15:13:48 2017

@author: li

please use "info_save_database_5views.py" to produce training samples in ./info_four_closest_corners_train_set
"""

from __future__ import print_function
import sys
sys.path.insert(0,'./models')
sys.path.insert(0,'./custom')
sys.path.insert(0,'./')
import argparse
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import custom_info_data5 as cd
from torchsample import transforms as tensor_tf
import glob
import numpy as np
from sklearn.cluster import KMeans
from Wgan_model_arbitrary import _netlocalD,_netlocalD2
from unet_full_cat3 import UNet as _netG
from tensorboardX import SummaryWriter
import time

writer = SummaryWriter()

model_dir='./model_para/'
parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=38, help='input batch size')
parser.add_argument('--imageSize', type=int, default=256, help='the height / width of the input image to network')

parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)

## Dnet  Unet
parser.add_argument('--nc', type=int, default=9)
parser.add_argument('--outch', type=int, default=3)
##

parser.add_argument('--niter', type=int, default=100000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda(1)', action='store_true', help='enables cuda(1)')
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


### Disparity Net
parser.add_argument('--input_ch', type=int,default=25)
parser.add_argument('--output_ch', type=int,default=1)
##

parser.add_argument('--train_size', type=int,default=3)
parser.add_argument('--train_label_dir',default='./info_four_closest_corners_train_set/')

opt = parser.parse_args()


cudnn.benchmark = True
def list_images1(folder, pattern='/*info', ext='npy'):
    filenames = sorted(glob.glob(folder + pattern + '.' + ext))
    return filenames

def list_images(folder, pattern='*.jpg'):
    filenames = sorted(glob.glob(folder + pattern))
    return filenames

train_label_feature_list=list_images1(opt.train_label_dir)


    
def DMAP_generation_BR(disparity,disparity1,disparity2):
    MAP=disparity+0
    MAP_B=disparity1+0
    MAP_R=disparity2+0
    d=(disparity.data).cpu().numpy()
    B_MAP=Variable(torch.zeros((disparity.size()))).cuda(1)
    R_MAP=Variable(torch.zeros((disparity.size()))).cuda(1)
    for i in range(0,d.shape[0]):
        ref=d[i,0,:,:]
        ind=np.where(ref!=0)
        km= KMeans(n_clusters=2, random_state=0).fit((ref[ind]).reshape(-1, 1) )

                
        TH1=np.amax(km.cluster_centers_)
        TH2=np.amin(km.cluster_centers_)
        coff1=0.2
        coff2=0.2
        mask_B=(MAP_B[i,0,:,:]>(TH1-(TH1-TH2)*coff1))*(MAP_B[i,0,:,:]!=0)
        mask_R=(MAP_R[i,0,:,:]<(TH2+(TH1-TH2)*coff2))*(MAP_R[i,0,:,:]!=0)

        B_MAP[i,0,:,:]=mask_B
        R_MAP[i,0,:,:]=mask_R
        

    return B_MAP.float(),R_MAP.float()
# data augmentation
data_transform1 = tensor_tf.Compose([
          tensor_tf.RandomFlip(),
      ])
    
data_transform2 = tensor_tf.Compose([
          tensor_tf.RandomFlip(),
      ])

affine_transform1=tensor_tf.RandomChoiceRotate([0,90,180,270])

affine_transform2=tensor_tf.RandomChoiceRotate([0,90,180,270])
train_set=cd.CustomDataset(train_label_feature_list,train_label_feature_list,
                           data_transform1=data_transform1,
                           data_transform2=data_transform2,
                           affine_transform1=affine_transform1,
                           affine_transform2=affine_transform2,
                           )


RdSpCrop = tensor_tf.RandomChoiceCompose([
          tensor_tf.SpecialCrop((opt.imageSize, opt.imageSize),0)
          ])

trainloader = torch.utils.data.DataLoader(train_set, batch_size=opt.train_size, 
                                          shuffle=True, num_workers=opt.workers)      

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        
netG = _netG(opt)
netG.apply(weights_init)


netD = _netlocalD(opt)
netD.apply(weights_init)

netD2 = _netlocalD2(opt)
netD2.apply(weights_init)


netD.cuda(1)
netD2.cuda(1)
netG.cuda(1)

    
overlapL2Weight = 10
wtl2=opt.wtl2


optimizerD1 = optim.RMSprop(netD.parameters(), lr = opt.lr)
optimizerD2 = optim.RMSprop(netD2.parameters(), lr = opt.lr)
optimizerG = optim.RMSprop(netG.parameters(), lr = opt.lr*10)
start_time = time.time()
for epoch in range(0,15000):

    netD.train()
    netD2.train()
    netG.train()
    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):

        ref,ref1_mag,ref2_mag,ref_mag,disparity,disparity1,disparity2,img1_color,img2_color,_= data
        
        batch_size = ref.size(0)

        ref=Variable(ref.cuda(1))
        img1_color=Variable(img1_color.cuda(1))
        img2_color=Variable(img2_color.cuda(1))
        # edge disparity for the overlapped image. Because the edges of two images are independent, we assume the edge disparity of two images can be additive
        disparity=Variable(disparity.cuda(1))   
        disparity1=Variable(disparity1.cuda(1))  # edge disparity for image1
        disparity2=Variable(disparity2.cuda(1))  # edge disparity for image2
        ref_mag=Variable(ref_mag.cuda(1))  # image gradient magtitude


        MMAP=(disparity!=0).float()
        DMAP_B,DMAP_R=DMAP_generation_BR(disparity,disparity1,disparity2)  #produce two thresholds
        DMAP_B,DMAP_R=DMAP_B*MMAP,DMAP_R*MMAP
        
        DMAP_B_GT=(disparity1!=0).float()*MMAP
        DMAP_R_GT=(disparity2!=0).float()*MMAP
        
        
        MMAP=(ref_mag>3).float()
        
        MAP1_F=DMAP_B*MMAP
        MAP2_F=DMAP_R*MMAP
        MAP_B_GT=DMAP_B_GT*MMAP
        MAP_R_GT=DMAP_R_GT*MMAP
        
        
        ref_M_B=ref*torch.cat((MAP1_F,MAP1_F,MAP1_F),dim=1)
        ref_M_R=ref*torch.cat((MAP2_F,MAP2_F,MAP2_F),dim=1)
        ref_M=ref*torch.cat((MMAP,MMAP,MMAP),dim=1)
        ref_M_B_GT=ref*torch.cat((MAP_B_GT,MAP_B_GT,MAP_B_GT),dim=1)
        ref_M_R_GT=ref*torch.cat((MAP_R_GT,MAP_R_GT,MAP_R_GT),dim=1)
        Input=torch.cat((ref_M,ref_M_B,ref_M_R),dim=1)
        

########################################################################
        #train discrinimator1
        netD.zero_grad()
        errD_real = netD(ref_M_B_GT)
        errD_real = errD_real.mean()
        D_x = errD_real.data.mean()
        fake = netG(Input)
        errD_fake = netD(fake.detach())
        errD_fake = errD_fake.mean()
        D_G_z1 = errD_fake.data.mean()
        errD = errD_fake - errD_real
        errD.backward()
        optimizerD1.step()
        
        for p in netD.parameters():
            p.data.clamp_(-0.01, 0.01)
 ########################################################################
        #train discrinimator2
        netD2.zero_grad()
        errD_real_overlap = netD2(ref_M_R_GT)
        errD_real_overlap = errD_real_overlap.mean()
        D_x_overlap = errD_real_overlap.data.mean()
        fake_overlap = ref_M -  fake
        errD_fake_overlap = netD2(fake_overlap.detach())
        errD_fake_overlap = errD_fake_overlap.mean()
        D_G_z1_overlap = errD_fake_overlap.data.mean()
        errD_overlap = errD_fake_overlap - errD_real_overlap
        errD_overlap.backward()
        optimizerD2.step()
        
        for p in netD2.parameters():
            p.data.clamp_(-0.01, 0.01)
 ########################################################################
         #train the generator
        netG.zero_grad()
        errG_D = netD(fake)
        errG_D = -errG_D.mean()
        fake_overlap = ref_M - fake
        errG_D_overlap = netD2(fake_overlap)
        errG_D_overlap = -errG_D_overlap.mean()
        
        
        wtl2Matrix = MAP1_F.clone()
        wtl2Matrix.data = wtl2Matrix.data*(wtl2*1 - wtl2)  +wtl2
        
        
        errG_l2 = (fake - ref_M_B_GT).pow(2) 
        errG_l2 = errG_l2 *  wtl2Matrix 
        errG_l2 = errG_l2.mean()
        
        errG = 7 * (1-wtl2) * errG_D + 7 * (1-wtl2) * errG_D_overlap  \
        + wtl2 * errG_l2 *14

        errG.backward()

        D_G_z2 = errG.data.mean()
        optimizerG.step()
        

        
    if epoch % 10 == 0:
        

        print('[%d/%d][%d/%d]  Loss_G: %.4f / %.4f  l1_1 %.4f l1_2 %.4f'
             % (epoch, opt.niter, i, len(trainloader),errG_D.data[0],errG_l2.data[0]))
