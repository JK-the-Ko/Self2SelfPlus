from torch import optim

from tqdm import tqdm

import config
from data import dataloaders
from models import models
from utils import utils


def main() : 
    # Read Options
    opt = config.readArguments()
    
    # Create DataLoader Instances
    imageSavePath, dataLoader = dataloaders.getDataLoaders(opt)
    
    # Create Save Directory
    utils.mkdirs(imageSavePath)
    
    for data in dataLoader :
        # Create Model Instance
        model = models.Self2SelfPlus(opt)
        model = models.assignOnGpu(opt, model)
        
        # Create Optimizer Instance
        optimizer = optim.Adam(model.net.parameters(), 
                               lr=opt.lr, 
                               betas=(opt.beta1, opt.beta2))

        # Load Data
        noisyImage, name = data["noisyImage"], data["name"]
        
        # Assign Device
        noisyImage = models.preprocessData(opt, noisyImage)
        
        # Get Image File Name
        print("================================================================================================================================")
        print(f"< Image File Name : {name[0]} >")
    
        with tqdm(total=opt.numIters) as pBar :
            for iter in range(1, opt.numIters+1) :
                # Train Denoising Autoencoder
                optimizer.zero_grad()
                loss = model(noisyImage, mode="train")
                
                # Compute Loss
                lossSS = loss["Self-Supervised"].item()
                lossIQA = loss["IQA"].item()
                
                # Back-Propagation
                loss = sum(loss.values()).mean()
                loss.backward()
                optimizer.step()

                # Show Training Procedure
                pBar.set_description(desc=f"[{iter}/{opt.numIters}] < Loss(Self-Supervised):{lossSS:.8f} | Loss(IQA):{lossIQA:.8f} >")
                pBar.update()

        # Get Inference Result
        denoisedImage = model(noisyImage, "inference")
        
        # Clamp Image
        denoisedImage = denoisedImage.clamp(0,1)
        
        # Save Image
        utils.saveImage(denoisedImage, imageSavePath, name[0])
        print()


if __name__ == "__main__" :
    main()