from os import listdir
from os.path import join

from PIL import Image

from natsort import natsorted

from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class PolyUDataset(Dataset) :
    def __init__(self, opt) :
        # Inheritance
        super(PolyUDataset, self).__init__()
        
        # Initialize Variables
        self.opt = opt
        self.noisyImageDataset = self.getPathList()
    
    def __getitem__(self, index) :
        # Load Data
        noisyImage = Image.open(join(self.noisyImageDataset[0], self.noisyImageDataset[1][index]))
        
        # Transform Data
        noisyImage = self.transforms(noisyImage)
        
        return {"noisyImage" : noisyImage, "name" : self.noisyImageDataset[1][index].replace("_real", "ours")}
    
    def __len__(self) :
        return len(self.noisyImageDataset[1])

    def getPathList(self) :
        # Get Absolute Parent Path of Dataset
        noisyImagePath = join(self.opt.dataRoot, "PolyU", "real")
        
        # Create List Instance for Adding Dataset Path
        noisyImagePathList = listdir(noisyImagePath)
        
        # Sort List Instance
        noisyImagePathList = natsorted(noisyImagePathList)
        
        # Create List Instance for Adding File Name
        noisyImageNameList = [imageName for imageName in noisyImagePathList if ".JPG" in imageName]
        
        return (noisyImagePath, noisyImageNameList)
    
    def transforms(self, image) :
        # Convert to PyTorch Tensor
        image = TF.to_tensor(image)

        return image