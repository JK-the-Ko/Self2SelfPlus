# Self2Self+: Single-Image Denoising with Self-Supervised Learning and Image Quality Assessment Loss
### [Paper](https://arxiv.org/abs/2307.10695) | [BibTex](#citation)
## Abstract
Recently, denoising methods based on supervised learning have exhibited promising performance. However, their reliance on external datasets containing noisy-clean image pairs restricts their applicability. To address this limitation, researchers have focused on training denoising networks using solely a set of noisy inputs. To improve the feasibility of denoising procedures, in this study, we proposed a single-image self-supervised learning method in which only the noisy input image is used for network training. Gated convolution was used for feature extraction and no-reference image quality assessment was used for guiding the training process. Moreover, the proposed method sampled instances from the input image dataset using Bernoulli sampling with a certain dropout rate for training. The corresponding result was produced by averaging the generated predictions from various instances of the trained network with dropouts. The experimental results indicated that the proposed method achieved state-of-the-art denoising performance on both synthetic and real-world datasets. This highlights the effectiveness and practicality of our method as a potential solution for various noise removal tasks.

## Denoising Process
- ### CBSD68 Dataset with AWGN of σ=50
![denoising](https://github.com/JK-the-Ko/Self2SelfPlus/assets/55126482/86984463-fc8b-4181-a705-ec7e25efc1e0)

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
