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
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import custom_info_data5 as cd
from torchsample import transforms as tensor_tf
import glob
import torchvision.models as models
import numpy as np
from sklearn.cluster import KMeans
from unet_full_cat3 import UNet as _net
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
parser.add_argument('--nc', type=int, default=6)
parser.add_argument('--outch', type=int, default=3)
parser.add_argument('--niter', type=int, default=100000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--net', default='', help="path to net (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--nBottleneck', type=int,default=4000,help='of dim for bottleneck of encoder')
parser.add_argument('--overlapPred',type=int,default=4,help='overlapping edges')
parser.add_argument('--nef',type=int,default=64,help='of encoder filters in first conv layer')
parser.add_argument('--wtl2',type=float,default=0.995,help='0 means do not use else use with this weight')
parser.add_argument('--wtlD',type=float,default=0.001,help='0 means do not use else use with this weight')
parser.add_argument('--input_ch', type=int,default=25)
parser.add_argument('--output_ch', type=int,default=1)
parser.add_argument('--train_size', type=int,default=3)
parser.add_argument('--train_label_dir',default='/media/li/HD/info_four_closest_corners_train_set/') #please use "info_save_database_5views.py" to produce training samples in ./info_four_closest_corners_train_set

opt = parser.parse_args()



cudnn.benchmark = True
opt.cuda=True
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
    disparity1=(disparity1.data).cpu().numpy()
    
    for i in range(0,disparity1.shape[0]):
        ref=disparity1[i,0,:,:]
        ind=np.where(ref!=0)
        km= KMeans(n_clusters=2, random_state=0).fit((ref[ind]).reshape(-1, 1) )

                
        TH1=np.amax(km.cluster_centers_)
        TH2=np.amin(km.cluster_centers_)
#        
        mask_B=(MAP_B[i,0,:,:]>(TH1-(TH1-TH2)*0.2))*(MAP_B[i,0,:,:]!=0)
        mask_R=(MAP_R[i,0,:,:]<(TH2+(TH1-TH2)*0.2))*(MAP_R[i,0,:,:]!=0)

        MAP_B[i,0,:,:]=mask_B
        MAP_R[i,0,:,:]=mask_R

        
        
    return MAP_B.float(),MAP_R.float()



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
        
net = _net(opt)
net.apply(weights_init)

net.cuda()

    
overlapL2Weight = 10
wtl2=opt.wtl2


    
vgg16 = models.vgg16_bn(pretrained=True)
subclass=nn.Sequential(*list(vgg16.children())[0])
vggf=nn.Sequential(*list(subclass.children())[0:13])
for param in vggf.parameters():
    param.requires_grad = False
vggf=vggf.cuda()

# setup optimizer
optimizerG = optim.RMSprop(net.parameters(), lr = opt.lr*10)

start_time = time.time()
for epoch in range(0,100000):
    net.train()
    for i, data in enumerate(trainloader, 0):
        ref,ref1_mag,ref2_mag,_,disparity,disparity1,disparity2,img1_color,img2_color,gray= data
        
        batch_size = ref.size(0)

        ref=Variable(ref.cuda())   # overlapped image
        img1_color=Variable(img1_color.cuda())
        img2_color=Variable(img2_color.cuda())
        disparity=Variable(disparity.cuda())
        disparity1=Variable(disparity1.cuda())
        disparity2=Variable(disparity2.cuda())
        DMAP_B,DMAP_R=DMAP_generation_BR(disparity,disparity1,disparity2)

        MMAP=(disparity!=0).float()
        DMAP_B=(disparity1!=0).float()*MMAP
        DMAP_R=(disparity2!=0).float()*MMAP
        
        MAP1_F=DMAP_B*MMAP        #   use groud truth masks for training
        MAP2_F=DMAP_R*MMAP

        ref_M_B=ref*torch.cat((MAP1_F,MAP1_F,MAP1_F),dim=1)
        ref_M_R=ref*torch.cat((MAP2_F,MAP2_F,MAP2_F),dim=1)

        Input=torch.cat((ref-ref_M_R,ref_M_B),dim=1)
        

        fake = net(Input)

########################################################################           
        

        net.zero_grad()
        wtl2Matrix = MAP1_F.clone()
        wtl2Matrix.data = wtl2Matrix.data*(wtl2*1 - wtl2)  +wtl2
        
        
        errG_l2 = (fake - img1_color).pow(2)
        errG_l2 = errG_l2 *  wtl2Matrix 
        errG_l2 = errG_l2.mean()
        
        mse=(fake*255 - img1_color*255).pow(2)
        mse=mse.mean()
        
        errG_vggf = (vggf(fake)-vggf(img1_color)).pow(2)  #VGG perceptual loss
        errG_vggf = errG_vggf.mean(dim=1)  
        errG_vggf = errG_vggf * 1
        errG_vggf = errG_vggf.mean()
        
        errG = wtl2 * errG_l2 *8+ wtl2 * errG_vggf*10


        errG.backward()

        D_G_z2 = errG.data.mean()
        optimizerG.step()

        
    if epoch % 10 == 0:

        print('[%d/%d][%d/%d] ED: errG_l2:  %.4f, vgg_loss: %.4f'
             % (epoch, opt.niter, i, len(trainloader),errG_l2.data[0],errG_vggf.data[0],))


    
    
