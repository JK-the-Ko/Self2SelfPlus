import torch
from torch import nn
from torch.nn import init

from tqdm import tqdm

from models.autoencoder import DAE
from models.loss import *

from utils import utils


class Self2SelfPlus(nn.Module) :
    def __init__(self, opt) :
        # Inheritance
        super(Self2SelfPlus, self).__init__()
        
        # Create Model Instance
        self.opt = opt
        self.OKBLUE, self.ENDC = utils.bcolors.OKBLUE, utils.bcolors.ENDC
        self.net = DAE(opt)

        # Create IQA Loss Instance
        self.criterionIQA = IQALoss()
        
        # Compute Number of Parameters
        self.computeNumParameter()
        
        # Initialize Model Parameters
        utils.fixSeed(opt.seed)
        self.initializeNetwork()

    def forward(self, noisyImage, mode) :
        if mode == "train" :
            self.net.train()
            loss = self.computeLoss(noisyImage)
            return loss
        elif mode == "inference" :
            self.net.eval()
            with torch.no_grad() :
                finalImage = 0
                with tqdm(total=self.opt.numSample) as pBar :
                    for iter in range(1, self.opt.numSample+1) :
                        _, denoisedImage = self.denoiseImage(noisyImage)
                        finalImage += denoisedImage
                        pBar.set_description(desc=f"[{iter}/{self.opt.numSample}] < Saving Result! >")
                        pBar.update()
            return finalImage/self.opt.numSample
        else :
            raise ValueError(f"{mode} is not supported")

    def loadCheckpoints(self, loadModel) :
        if loadModel :
            self.net = utils.loadNetwork(self.opt, self.net)
            print("Model Loaded!")
        
    ############################################################################
    # Private helper methods
    ############################################################################

    def computeNumParameter(self) :
        networkList = [self.net]
        print(f"{self.OKBLUE}Self2Self+{self.ENDC}: Now Computing Model Parameters.")
        for network in networkList :
            numParameter = 0
            for _, module in network.named_modules() :
                if isinstance(module, nn.Conv2d) :
                    numParameter  += sum([p.data.nelement() for p in module.parameters()])
            print(f"{self.OKBLUE}Self2Self+{self.ENDC}: {utils.bcolors.OKGREEN}[{network.__class__.__name__}]{self.ENDC} Total params : {numParameter:,}.")
        print(f"{self.OKBLUE}Self2Self+{self.ENDC}: Finished Computing Model Parameters.")

    def initializeNetwork(self) :
        def init_weights(m, initType=self.opt.initType, gain=0.02) :
            className = m.__class__.__name__
            # Initialize Convolution and Linear Weights
            if hasattr(m, "weight") and className.find("Conv") != -1 :
                if initType == "normal" :
                    init.normal_(m.weight.data, 0.0, gain)
                elif initType == "xavier" :
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif initType == "xavier_uniform" :
                    init.xavier_uniform_(m.weight.data, gain=gain)
                elif initType == "kaiming" :
                    init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
                elif initType == "orthogonal" :
                    init.orthogonal_(m.weight.data, gain=gain)
                elif initType == "none" :
                    m.reset_parameters()
                else:
                    raise NotImplementedError(f"{initType} method is not supported")
                if hasattr(m, "bias") and m.bias is not None :
                    init.constant_(m.bias.data, 0.0)

        # Create List Instance for Adding Network
        networkList = [self.net]
        
        #Initialize Network Weights
        for network in networkList :
            network.apply(init_weights)

    def computeLoss(self, noisyImage) :
        # Create Dictionary Instance for Adding Loss
        loss = {}
        
        # Get Inference Result
        mask, denoisedImage = self.denoiseImage(noisyImage)
        
        # Compute Loss
        if self.opt.lossType == "L1" :
            loss["N2N"] = torch.abs((denoisedImage-noisyImage)*(1-mask)).sum()/(1-mask).sum()
        elif self.opt.lossType == "L2" :
            loss["N2N"] = torch.pow((denoisedImage-noisyImage)*(1-mask), 2).sum()/(1-mask).sum()
        
        # Compute IQA Loss
        loss["IQA"] = self.criterionIQA(denoisedImage)*self.opt.lambdaIQA
        
        return loss
            
    def denoiseImage(self, noisyImage) :
        # Get Inference Result
        return self.net(noisyImage)


def assignOnGpu(opt, model):
    if opt.gpuIds != "-1":
        model = model.cuda()
    
    return model


def preprocessData(opt, noisyImage) :
    if opt.gpuIds != "-1" :
        noisyImage = noisyImage.cuda()
        
    return noisyImage