import pyiqa

import torch
from torch import nn


class IQALoss(nn.Module) :
    def __init__(self) :
        # Inheritance
        super(IQALoss, self).__init__()
        
        # Create IQA Model Instance
        self.model = pyiqa.create_metric("paq2piq", as_loss=True)
        
    def forward(self, input) :
        # Compute Loss
        return torch.pow(100-self.model(input), 2).mean()