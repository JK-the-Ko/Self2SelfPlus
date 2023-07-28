# Self2Self+: Single-Image Denoising with Self-Supervised Learning and Image Quality Assessment Loss
### [Paper](https://arxiv.org/abs/2307.10695) | [BibTex](#citation)
## Abstract
Recently, denoising methods based on supervised learning have exhibited promising performance. However, their reliance on external datasets containing noisy-clean image pairs restricts their applicability. To address this limitation, researchers have focused on training denoising networks using solely a set of noisy inputs. To improve the feasibility of denoising procedures, in this study, we proposed a single-image self-supervised learning method in which only the noisy input image is used for network training. Gated convolution was used for feature extraction and no-reference image quality assessment was used for guiding the training process. Moreover, the proposed method sampled instances from the input image dataset using Bernoulli sampling with a certain dropout rate for training. The corresponding result was produced by averaging the generated predictions from various instances of the trained network with dropouts. The experimental results indicated that the proposed method achieved state-of-the-art denoising performance on both synthetic and real-world datasets. This highlights the effectiveness and practicality of our method as a potential solution for various noise removal tasks.

## Denoising Process
- ### CBSD68 Dataset with AWGN of σ=50
<img src="gifs/denoising.gif">

## Prerequisites
- Python 3.8.10
- PyTorch>=1.12.1
- Torchvision>=0.13.1
- NVIDIA GPU + CUDA cuDNN

## Installation
- ### Clone this repo.
```
git clone https://github.com/JK-the-Ko/Self2SelfPlus.git
cd Self2SelfPlus/
```
- ### Install PyTorch and dependencies from http://pytorch.org
- ### Please install dependencies by
```
pip install -r requirements.txt
```

## Dataset
- CBSD68 dataset with AWGN of 15, 25, and 50. The following dataset should be placed in ```dataset/CBSD68/sigma-N``` folder.
- SIDD dataset. The following dataset should be placed in ```dataset/SIDD/Noisy``` folder.
- PolyU dataset. The following dataset should be placed in ```dataset/PolyU/real``` folder.

## Training
The following script is for training **CBSD68 dataset with different AWGNs**. We recommend using commands written in the scripts folder.
```
python train.py --dataType CBSD68 --sigma AWGN-SIGMA --p 0.4 --numIters 4000
```
The following script is for training **SIDD dataset**. We recommend using commands written in the scripts folder.
```
python train.py --dataType SIDD --p 0.9 --numIters 1000
```
The following script is for training **PolyU dataset**. We recommend using commands written in the scripts folder.
```
python train.py --dataType PolyU --p 0.7 --numIters 5000
```

## Evaluation
During training, the final result will be saved automatically in ```results/dataset-name``` folder.

## Pre-Trained Models
Since it is a single-image self-supervised learning task, **no pre-trained models can be given**.

## Citation
If you use **Self2Self+** in your work, please consider citing us as

```
@misc{ko2023self2self,
      title={Self2Self+: Single-Image Denoising with Self-Supervised Learning and Image Quality Assessment Loss}, 
      author={Jaekyun Ko and Sanghwan Lee},
      year={2023},
      eprint={2307.10695},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
