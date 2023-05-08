import importlib
from torch.utils.data import DataLoader


def getDataLoaders(opt) :
    # Select Dataset Name
    if opt.dataType == "CBSD68" :
        datasetName = "CBSD68Dataset"
        imageSavePath = f"{opt.resultsDir}/{opt.name}/{opt.dataType}/sigma-{opt.sigma}"
    elif opt.dataType == "SIDD" :
        datasetName = "SIDDDataset"
        imageSavePath = f"{opt.resultsDir}/{opt.name}/{opt.dataType}"
    elif opt.dataType == "PolyU" :
        datasetName = "PolyUDataset"
        imageSavePath = f"{opt.resultsDir}/{opt.name}/{opt.dataType}"
    else :
        raise ValueError(f"{opt.dataType} is not supported")

    # Import Python Code
    fileName = importlib.import_module(f"data.{datasetName}")
    
    # Create Dataset Instance
    dataset = fileName.__dict__[datasetName](opt)
    
    # Train PyTorch DataLoader Instance
    dataLoader = DataLoader(dataset, 
                            batch_size=1, 
                            shuffle=False, 
                            drop_last=False, 
                            num_workers=opt.numWorkers)
    
    
    return imageSavePath, dataLoader