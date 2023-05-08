import argparse


def readArguments() :
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser = addAllArguments(parser)
    opt = parser.parse_args()
    
    return opt


def addAllArguments(parser):
    # General options
    parser.add_argument("--name", type=str, default="Self2Self+", help="name of the experiment. It decides where to store samples and models")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--gpuIds", type=str, default="0", help="use 0 for gpu, -1 for CPU")
    parser.add_argument("--dataRoot", type=str, default="dataset/", help="path to dataset root")
    parser.add_argument("--dataType", type=str, required=True, help="data type type for training, choices from {CBSD68 | SIDD}")
    parser.add_argument("--sigma", type=int, default=25, help="standard deviation of synthetic noise distribution, choices from {15 | 25 | 50}")
    parser.add_argument("--p", type=float, default=0.4, help="probability of dropout")
    parser.add_argument("--numWorkers", type=int, default=0, help="num_workers argument for dataloader")

    # For denoising autoencoder
    parser.add_argument("--inputDim", type=int, default=3, help="# channels of input noisy image")
    parser.add_argument("--channels", type=int, default=48, help="# of filters in first conv layer in auto-encoder")
    parser.add_argument("--initType", type=str, default="normal", help="selects weight initialization type")
    parser.add_argument("--numSample", type=int, default=500, help="number of bernoulli samples for inference")

    # For training
    parser.add_argument("--numIters", type=int, default=4000, help="number of epochs to train")
    parser.add_argument("--beta1", type=float, default=0.9, help="momentum term of adam")
    parser.add_argument("--beta2", type=float, default=0.999, help="momentum term of adam")
    parser.add_argument("--lr", type=float, default=4e-4, help="learning rate, default=0.0004")
    parser.add_argument("--lossType", type=str, default="L1", help="loss type for training, choices from {L1 | L2}")
    parser.add_argument("--lambdaIQA", type=float, default=2e-8, help="weight of style loss")
    
    # For inference
    parser.add_argument("--resultsDir", type=str, default="./results/", help="saves testing results here.")

    return parser