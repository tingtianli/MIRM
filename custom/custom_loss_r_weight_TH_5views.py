#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 18:30:52 2017

@author: li
"""

from torch.autograd import Variable
import torch
from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models

def gradx(input):

    filter_dx=torch.autograd.Variable(torch.zeros(input.size()[1],input.size()[1],1,2))
    dx=torch.from_numpy(np.array([-1,1],np.float32))
#    for k1 in range(0,input.size()[1]):
#        for k2 in range(0,input.size()[1]):
    filter_dx[0,0,0,:]=torch.autograd.Variable(dx) 
#    filter_dx[1,1,0,:]=torch.autograd.Variable(dx)  
#    filter_dx[2,2,0,:]=torch.autograd.Variable(dx)  
    filter_dx=filter_dx.cuda(0)
    result=F.conv2d(input,filter_dx,padding=(0,1))
    
        
    return result
    
def grady(input):

    filter_dy=torch.autograd.Variable(torch.zeros(input.size()[1],input.size()[1],2,1))
    dy=torch.from_numpy(np.array([[-1],[1]],np.float32))
#    for k1 in range(0,input.size()[1]):
#        for k2 in range(0,input.size()[1]):
    filter_dy[0,0,:,:]=torch.autograd.Variable(dy)
#    filter_dy[1,1,:,:]=torch.autograd.Variable(dy)
#    filter_dy[2,2,:,:]=torch.autograd.Variable(dy)
    filter_dy=filter_dy.cuda(0)
    result=F.conv2d(input,filter_dy,padding=(1,0))
    
        
    return result

def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as volatile or not requiring gradients"


class _Loss(Module):
    def __init__(self, size_average=True):
        super(_Loss, self).__init__()
        self.size_average = size_average


class _WeightedLoss(_Loss):
    def __init__(self, weight=None, size_average=True):
        super(_WeightedLoss, self).__init__(size_average)
        self.register_buffer('weight', weight)
        
def build_stack(original,ind,length):
    target1=original[:,ind,:,:]
    target2=target1.unsqueeze(1)
    target3=target2.repeat(1,length,1,1)
    return target3

        
class CustomLoss(_Loss):
    r"""Creates a criterion that measures the mean squared error between
    `n` elements in the input `x` and target `y`:

    :math:`{loss}(x, y)  = 1/n \sum |x_i - y_i|^2`

    `x` and `y` arbitrary shapes with a total of `n` elements each.

    The sum operation still operates over all the elements, and divides by `n`.

    The division by `n` can be avoided if one sets the internal variable
    `size_average` to `False`.

    Args:
        size_average (bool, optional): By default, the losses are averaged
           over observations for each minibatch. However, if the field
           size_average is set to False, the losses are instead summed for
           each minibatch. Default: True

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Target: :math:`(N, *)`, same shape as the input

    Examples::

        >>> loss = nn.MSELoss()
        >>> input = autograd.Variable(torch.randn(3, 5), requires_grad=True)
        >>> target = autograd.Variable(torch.randn(3, 5))
        >>> output = loss(input, target)
        >>> output.backward()
    """
    def forward(self, original,disparity,mag):
#        center_used=3
        
        selected=np.array([0,1])
  
#        grid=Variable(torch.zeros([original.size()[0],original.size()[2],original.size()[3],2])).cuda(1)
        loss=0
        target_overlap0=build_stack(original,0,1)
        target_overlap1=build_stack(original,1,1)
        target_overlap2=build_stack(original,2,1)
        target_overlap3=build_stack(original,3,1)
        target_overlap4=build_stack(original,4,1)
        target_mag0=build_stack(mag,0,1)
        target_mag1=build_stack(mag,1,1)
        target_mag2=build_stack(mag,2,1)
        target_mag3=build_stack(mag,3,1)
        target_mag4=build_stack(mag,4,1)
#        r_weight=r_weight.repeat(1,length*length,1,1)
#        mag=mag.repeat(1,length*length,1,1)
        
        w=original.size()[3]
        h=original.size()[2]
        disp=disparity[:,0,:,:]

        theta=torch.FloatTensor([[1,0,0],[0,1,0]])
        theta=theta.unsqueeze(0)
        theta=theta.repeat(original.size()[0],1,1)
        
        theta=Variable(theta)
#        th=15/255.
        grid_ori=torch.nn.functional.affine_grid(theta, torch.Size((original.size()[0],1, h, w)))
        grid_ori=grid_ori.cuda(0)


        grid_shift0=torch.stack((-1*disp/w/2,            #5_5
                                -1*disp/h/2),dim=3)
        grid1=grid_ori+grid_shift0
        warped_ori=F.grid_sample(target_overlap0, grid1, mode='bilinear')#, padding_mode='border')
        warped_ori_w=warped_ori*target_mag1
        target_overlap_w=target_overlap1*target_mag1
        diff1=F.mse_loss(warped_ori_w, target_overlap_w)
        
        
        grid_shift0=torch.stack((1*disp/w/2,             #4_4
                                -1*disp/h/2),dim=3)
        grid1=grid_ori+grid_shift0
        warped_ori=F.grid_sample(target_overlap0, grid1, mode='bilinear')#, padding_mode='border')
        warped_ori_w=warped_ori*target_mag2
        target_overlap_w=target_overlap2*target_mag2
        diff2=F.mse_loss(warped_ori_w, target_overlap_w)
        
        
        grid_shift0=torch.stack((-1*disp/w/2,                #7_7
                                1*disp/h/2),dim=3)
        grid1=grid_ori+grid_shift0
        warped_ori=F.grid_sample(target_overlap0, grid1, mode='bilinear')#, padding_mode='border')
        warped_ori_w=warped_ori*target_mag3
        target_overlap_w=target_overlap3*target_mag3
        diff3=F.mse_loss(warped_ori_w, target_overlap_w)
        

        grid_shift0=torch.stack((1*disp/w/2,                   #8_8
                                 1*disp/h/2),dim=3)
        grid1=grid_ori+grid_shift0
        warped_ori=F.grid_sample(target_overlap0, grid1, mode='bilinear')#, padding_mode='border')
        warped_ori_w=warped_ori*target_mag4
        target_overlap_w=target_overlap4*target_mag4
        diff4=F.mse_loss(warped_ori_w, target_overlap_w)
        
        dx=gradx(disparity)[:,:,0:128,0:128]
        dy=grady(disparity)[:,:,0:128,0:128]
        
        TV=F.l1_loss(torch.abs(dx)+torch.abs(dy),0*disparity)
        loss= diff1+diff2+diff3+diff4#+0.000002*TV
                

                
        return loss




