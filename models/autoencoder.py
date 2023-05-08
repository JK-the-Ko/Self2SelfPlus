import torch
from torch import nn
import torch.nn.functional as F

from models.gatedConv import GatedConv2d
from models.architecture import Downscale2d, Upscale2d


class DAE(nn.Module) :
    def __init__(self, opt) :
        # Inheritance
        super(DAE, self).__init__()
        
        # Initialize Variable
        self.opt = opt
        channels = opt.channels

        # Create Encoder Layer Instance
        self.EB0 = nn.Sequential(GatedConv2d(opt.inputDim*2, channels, kernelSize=3, stride=1, padding=1),
                                 GatedConv2d(channels, channels, kernelSize=3, stride=1, padding=1))
        self.EB1 = Downscale2d(channels)
        self.EB2 = Downscale2d(channels)
        self.EB3 = Downscale2d(channels)
        self.EB4 = Downscale2d(channels) 
        self.EB5 = Downscale2d(channels)
        
        # Create Decoder Layer Instance
        self.DB0 = Upscale2d(opt, channels, channels, channels*2)
        self.DB1 = Upscale2d(opt, channels*2, channels, channels*2)
        self.DB2 = Upscale2d(opt, channels*2, channels, channels*2)
        self.DB3 = Upscale2d(opt, channels*2, channels, channels*2)
        self.DB4 = Upscale2d(opt, channels*2, channels, channels*2)
        self.DB5 = nn.Conv2d(channels*2, opt.inputDim, kernel_size=3, stride=1, padding=1, padding_mode="reflect")

    def forward(self, input) :
        # Get Tensor Size
        n, c, h, w = input.size()
        
        # Generate Bernoulli Mae
        mask = F.dropout(torch.ones((n,c,h,w), device=input.device), 
                         self.opt.p, True)*(1-self.opt.p)
        
        # Sample Image
        input = (mask*input).detach()
        
        # Encoder
        e0 = self.EB0(torch.cat([mask, input], dim=1)) # H x W x C
        e1 = self.EB1(e0) # H/2 x W/2 x C
        e2 = self.EB2(e1) # H/4 x W/4 x C
        e3 = self.EB3(e2) # H/8 x W/8 x C
        e4 = self.EB4(e3) # H/16 x W/16 x C
        e5 = self.EB5(e4) # H/32 x W/32 x C
        
        # Decoder
        d1 = self.DB0(e5, e4) # H/16 x W/16 x C
        d2 = self.DB1(d1, e3) # H/8 x W/8 x C
        d3 = self.DB2(d2, e2) # H/4 x W/4 x C
        d4 = self.DB3(d3, e1) # H/2 x W/2 x C
        d5 = self.DB4(d4, e0) # H x W x C
        d6 = self.DB5(d5)
            
        return mask, d6