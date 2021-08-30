#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 15:13:48 2017

@author: li

please use "npy_save_database_5views.py" to produce training samples in ./npy_all_four_closest_corners/
"""

from __future__ import print_function
import sys
sys.path.insert(0,'./models')
sys.path.insert(0,'./custom')
import argparse
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import Custom_data_disparity as cd
from torchsample import transforms as tensor_tf
import glob
from custom_loss_r_weight_TH_5views import CustomLoss
from pure_conv import Disparity_Net
model_dir='./model_para/'

parser = argparse.ArgumentParser()
parser.add_argument('--imageSize', type=int,default=64)
parser.add_argument('--train_size', type=int,default=8)
parser.add_argument('--workers', type=int,default=8)
parser.add_argument('--input_ch', type=int,default=5)
parser.add_argument('--output_ch', type=int,default=1)
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--lr', type=float, default=0.00002, help='learning rate, default=0.0001')
parser.add_argument('--Model_pkl',default=model_dir+'Model_Disparity.pkl')
parser.add_argument('--train_label_dir',default='./npy_all_four_closest_corners/') #please use "npy_save_my_database_5views.py" to produce training samples
opt = parser.parse_args()
print(opt)


cudnn.benchmark = True
opt.cuda=True
def list_images1(folder, pattern='/*_sub_imgs', ext='npy'):
    filenames = sorted(glob.glob(folder + pattern + '.' + ext))
    return filenames

def list_images(folder, pattern='*'):
    filenames = sorted(glob.glob(folder + pattern))
    return filenames


train_label_feature_list=list_images1(opt.train_label_dir)
    
## data augmentation
data_transform1 = tensor_tf.Compose([
          tensor_tf.RandomFlip(),
      ])
    
data_transform2 = tensor_tf.Compose([
          tensor_tf.RandomFlip(),
      ])
affine_transform=tensor_tf.AffineCompose([
        tensor_tf.RandomAffine(
                 rotation_range=10, 
                 translation_range=None,
                 shear_range=10, 
                 zoom_range=(0.9,1.1),
                 interp='bilinear',
                 lazy=False
                  )
        ]
        )
affine_transform1=tensor_tf.RandomChoiceRotate([0,90,180,270])
affine_transform2=tensor_tf.RandomChoiceRotate([0,90,180,270])   
train_set=cd.CustomDataset(train_label_feature_list,
                           data_transform1=data_transform1,
                           data_transform2=data_transform2,
                           affine_transform1=affine_transform1,
                           affine_transform2=affine_transform2
                           )

RdSpCrop = tensor_tf.RandomChoiceCompose([
          tensor_tf.SpecialCrop((opt.imageSize, opt.imageSize),0)
          ])

trainloader = torch.utils.data.DataLoader(train_set, batch_size=opt.train_size, 
                                          shuffle=True, num_workers=opt.workers)      
     
Model = Disparity_Net(opt)
# Model.load_state_dict(torch.load(opt.Model_pkl))
criterion = CustomLoss()


if opt.cuda:
    Model.cuda()
    criterion.cuda()

# setup optimizer
optimizer = optim.Adam(Model.parameters(), lr=opt.lr/30, betas=(0.9, 0.999))

for epoch in range(0,100000):
    Model.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        LF,mag,LF1,LF2 = data
        batch_size = LF.size(0)

        mag=Variable(mag.cuda())
        LF=Variable(LF.cuda())
        LF1=Variable(LF1.cuda())
        LF2=Variable(LF2.cuda())

        optimizer.zero_grad() 
        disparity = Model(LF)
        loss = criterion(LF,disparity,mag)  # gradient emphasized loss function
        loss.backward() 
        optimizer.step() 
        running_loss += loss.data[0]  
        if i % 50 == 0 and i!=0: 
            print('[%d, %5d] train loss: %.5f' % (epoch, i,  2000*running_loss/50.))
            running_loss = 0.0

            

    
    
    
    
    
