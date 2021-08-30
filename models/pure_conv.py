import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def gradx(input):
#    batch_num=int(input.size(0))
    filter_dx=torch.autograd.Variable(torch.randn(1,1,1,3)).cuda(1)
    dx=torch.from_numpy(np.array([0,-1,1],np.float32)).cuda(1)
    filter_dx[0,0,:,:]=torch.autograd.Variable(dx)
    result=F.conv2d(input,filter_dx,padding=(0,1))
    return result
    
def grady(input):
#    batch_num=int(input.size(0))
    filter_dy=torch.autograd.Variable(torch.randn(1,1,3,1)).cuda(1)
    dy=torch.from_numpy(np.array([[0],[-1],[1]],np.float32)).cuda(1)
    filter_dy[0,0,:,:]=torch.autograd.Variable(dy)
    result=F.conv2d(input,filter_dy,padding=(1,0))
    return result
    

class Disparity_Net(nn.Module):
    def __init__(self, opt):
        super(Disparity_Net, self).__init__()
        self.ngpu = opt.ngpu
        self.main = nn.Sequential(
            # input is (nc) x 128 x 128
#            nn.Conv2d(opt.input_ch,256,kernel_size=5, stride=1, padding=2, bias=False),
##            nn.LeakyReLU(0.2, inplace=True),
#            nn.BatchNorm2d(256),
            
#            nn.Conv2d(opt.input_ch,128,kernel_size=5, stride=1, padding=2, bias=False),
#            nn.BatchNorm2d(128),
#            nn.ReLU(inplace=True),
            
            nn.Conv2d(opt.input_ch,256,kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
#            nn.PReLU(1),
#            nn.LeakyReLU(0.2, inplace=True),
            
            # state size: (nef) x 64 x 64
            nn.Conv2d(256,128,kernel_size=5, stride=1, padding=2, bias=False),
#            nn.Conv2d(128,128,kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # state size: (nef) x 32 x 32
            nn.Conv2d(128,128,kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(128),
#            nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(inplace=True),
#            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(128),
#            nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128,128,kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(128),
#            nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128,128,kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(128),
#            nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(inplace=True),            
            # state size: (nef*2) x 16 x 16
            nn.Conv2d(128,128,kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(128),
#            nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128,opt.output_ch,kernel_size=5, stride=1, padding=2, bias=False),

            
#            opt.nef
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
            
#        output_dx=gradx(output)
#        output_dy=grady(output)
#        final=torch.cat((output,output_dx,output_dy),1)
        return output

