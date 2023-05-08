import torch
from torch import nn
import torch.nn.functional as F

from models.gatedConv import GatedConv2d


class Downscale2d(nn.Module) :
    def __init__(self, channels) :
        # Inheritance
        super(Downscale2d, self).__init__()
        
        # Create Convolution Layer Instance
        self.conv0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = GatedConv2d(channels, channels, kernelSize=3, stride=1, padding=1)
        
    def forward(self, input) :
        output = self.conv0(input)
        output = self.conv1(output)
        
        return output


class Upscale2d(nn.Module) :
    def __init__(self, opt, inChannels, skChannels, outChannels) :
        # Inheritance
        super(Upscale2d, self).__init__()
        
        self.opt = opt
        
        # Create Convolution Layer Instance
        self.conv0 = nn.Sequential(nn.Conv2d(inChannels+skChannels, outChannels, kernel_size=3, stride=1, padding=1, padding_mode="reflect"),
                                   nn.LeakyReLU(0.2))
        self.conv1 = nn.Sequential(nn.Conv2d(outChannels, outChannels, kernel_size=3, stride=1, padding=1, padding_mode="reflect"),
                                   nn.LeakyReLU(0.2))
        
    def forward(self, input, skipConnection) :
        output = torch.cat([F.interpolate(input, scale_factor=2, mode="nearest"), 
                            skipConnection], dim=1)

        output = self.conv0(F.dropout(output, self.opt.p, training=True))
        output = self.conv1(F.dropout(output, self.opt.p, training=True))
        
        return output