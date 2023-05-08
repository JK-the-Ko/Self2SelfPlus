from os import listdir
from os.path import join

from PIL import Image

from natsort import natsorted

from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class CBSD68Dataset(Dataset) :
    def __init__(self, opt) :
        # Inheritance
        super(CBSD68Dataset, self).__init__()
        
        # Initialize Variables
        self.opt = opt
        self.noisyImageDataset = self.getPathList()
    
    def __getitem__(self, index) :
        # Load Data
        noisyImage = Image.open(join(self.noisyImageDataset[0], self.noisyImageDataset[1][index]))
        
        # Transform Data
        noisyImage = self.transforms(noisyImage)
        
        return {"noisyImage" : noisyImage, "name" : self.noisyImageDataset[1][index]}
    
    def __len__(self) :
        return len(self.noisyImageDataset[1])

    def getPathList(self) :
        # Get Absolute Parent Path of Dataset
        noisyImagePath = join(self.opt.dataRoot, "CBSD68", f"sigma-{self.opt.sigma}")
        
        # Create List Instance for Adding Dataset Path
        noisyImagePathList = listdir(noisyImagePath)
        
        # Sort List Instance
        noisyImagePathList = natsorted(noisyImagePathList)
        
        # Create List Instance for Adding File Name
        noisyImageNameList = [imageName for imageName in noisyImagePathList if ".png" in imageName]
        
        return (noisyImagePath, noisyImageNameList)
    
    def transforms(self, image) :
        # Get Image Size
        width, height = image.size
        
        # Resize Image
        image = TF.resize(image, 
                          size=(height-1, width-1), 
                          interpolation=TF.InterpolationMode.NEAREST)

        # Convert to PyTorch Tensor
        image = TF.to_tensor(image)

        return image