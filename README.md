# mmrotate-dcfl
Official implementation for the CVPR23 paper: Dynamic Coarse-to-Fine Learning for Oriented Tiny Object Detection.
Coming soon!

## Introduction
DCFL is a learning framework for detecting oriented tiny objects.

**Abstract**:     Detecting arbitrarily oriented tiny objects poses intense challenges to existing detectors, especially for the label assignment. Despite the exploration of adaptive label assignment in recent oriented object detectors, the extreme geometry shape and limited feature of oriented tiny objects still induce severe mismatch and imbalance issues. Specifically, the position prior, positive sample feature, and instance are mismatched, and the learning of extreme-shaped objects is biased and unbalanced due to little proper feature supervision. To tackle these issues, we propose a dynamic prior along with the coarse-to-fine assigner, dubbed DCFL. For one thing, we model the prior, label assignment, and object representation all in a dynamic manner to alleviate the mismatch issue. For another, we leverage the coarse prior matching and finer posterior constraint to dynamically assign labels, providing appropriate and relatively balanced supervision for diverse instances. Extensive experiments on six datasets show substantial improvements to the baseline. Notably, we obtain the state-of-the-art performance for one-stage detectors on the DOTA-v1.5, DOTA-v2.0, and DIOR-R datasets under single-scale training and testing.

![demo image](figures/framework_final.png)

## Installation and Get Started

Required environments:
- Linux
- Python 3.7+
- PyTorch 1.10.0+
- CUDA 9.2+
- GCC 5+
- MMdet 2.23.0+
- [MMCV-DCFL]() 


Install:
Note that this repository is based on the MMRotate. Assume that your environment has satisfied the above requirements, please follow the following steps for installation.

```
git clone https://github.com/Chasel-Tsui/mmrotate-dcfl.git
cd mmrotate-dcfl
pip install -r requirements/build.txt
python setup.py develop
```

## Main Results

DOTA-v1.0

| Method |         Backbone         | AP50  | AP75  | mAP | Angle | lr schd | Aug  | Batch Size |                           Configs                          |  
| :-----: | :----------------------: | :---: | ----- | :---: | :-----: | :--: | :-------: |:-----:| :----------------------------------------------------------: | 
|RetinaNet-O| ResNet50 (1024,1024,200) | 69.17 | 36.71 | 39.33| le135  |   1x    |  Flipping   |     2      | [retinanet_obb_r50_dota1]() | 
|R3Det| ResNet50 (1024,1024,200) | 70.18 | 35.54 | 37.73| oc  |   1x    |  Flipping   |     2      | [r3det_oc_r50_dota1]() |
|ATSS-O| ResNet50 (1024,1024,200) | 73.12 | 43.66 | 43.04| le135  |   1x    |  Flipping   |     2      | [atss_le135_r50_dota1]() |
|S2A-Net| ResNet50 (1024,1024,200) | 74.12 | 43.14 | 42.33| le90  |   1x    |  Flipping   |     2      | [s2a_le90_r50_dota1]() | 
|DCFL| ResNet50 (1024,1024,200) | **74.26** | **47.55** | **45.07** | le135  |   1x    |  Flipping   |     2      |     [dcfl_r50_dota1]()      | 

DOTA-v2.0

| Method |         Backbone         | AP50  |  Angle | lr schd | Aug  | Batch Size |                           Configs                          | Speed |
| :-----: | :----------------------: | :---: | :-----: | :--: | :-------: |:-----:| :----------------------------------------------------------: | :--: |
|RetinaNet-O| ResNet50 (1024,1024,200) | 46.68 |  le135  |   1x    |  Flipping   |     2      | [retinanet_obb_r50_dota2]() | 20.8 FPS|
|R3Det w/ KLD| ResNet50 (1024,1024,200) | 47.26 |  le135  |   1x    |  Flipping   |     2      | [r3det_oc_r50_dota2]() | 16.2 FPS |
|ATSS-O| ResNet50 (1024,1024,200) | 49.57 |  le135  |   1x    |  Flipping   |     2      | [atss_le135_r50_dota2]() | - |
|S2A-Net| ResNet50 (1024,1024,200) | 49.86 |  le135  |   1x    |  Flipping   |     2      | [s2a_le135_r50_dota2]() | 18.9 FPS|
|DCFL| ResNet50 (1024,1024,200) | 51.57 | le135  |   1x    |  Flipping   |     2      |     [dcfl_r50_dota2]()      | 20.9 FPS |
|DCFL| ResNet101 (1024,1024,200) | **52.54** | le135  |   1x    |  Flipping   |     2      |     [dcfl_r101_dota2]()      | - |

## Visualization


## Citation
If you find this work helpful, please consider citing:
```bibtex
```