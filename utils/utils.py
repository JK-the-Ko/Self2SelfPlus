from os import makedirs
from os.path import join, exists

import random

import cv2
import numpy as np

import torch


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def fixSeed(seed) :
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count


def tensorToImage(tensor) :
    imageNumpy = np.transpose((tensor*255).detach().cpu().numpy(), 
                              (1, 2, 0)).astype("uint8")
    return imageNumpy


def saveImage(tensor, imageSavePath, name) :
    image = tensorToImage(tensor[0,:,:,:])
    cv2.imwrite(join(imageSavePath, name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def mkdirs(path) :
    # Make Directory
    if not exists(path) :
        makedirs(path)