# PICASO
Official PyTorch implemetation for the paper [PICASO:Permutation-Invariant Cascaded Attentive Set Operator](https://arxiv.org/abs/2107.08305).

## Requirements
* Python 3
* torch >= 1.0
* numpy
* matplotlib
* scipy
* tqdm

## Abstract
Set-input deep networks have recently drawn much interest in computer vision and machine learning. This is in part due to the increasing number of important tasks such as meta-learning, clustering, and anomaly detection that are defined on set inputs. These networks must take an arbitrary number of input samples and produce the output invariant to the input set permutation. Several algorithms have been recently developed to address this urgent need. Our paper analyzes these algorithms using both synthetic and real-world datasets, and shows that they are not effective in dealing with common data variations such as image translation or viewpoint change. To address this limitation, we propose a permutation-invariant cascaded attentional set operator (PICASO). The gist of PICASO is a cascade of multihead attention blocks with dynamic templates. The proposed operator is a stand-alone module that can be adapted and extended to serve different machine learning tasks. We demonstrate the utilities of PICASO in four diverse scenarios: (i) clustering, (ii) image classification under novel viewpoints, (iii) image anomaly detection, and (iv) state prediction. PICASO increases the SmallNORB image classification accuracy with novel viewpoints by about 10% points. For set anomaly detection on CelebA dataset, our model improves the areas under ROC and PR curves dataset by about 22% and 10%, respectively. For the state prediction on CLEVR dataset, it improves the AP by about 40%.

## Experiments
This repository implements the amortized clustering, classification, set anomaly detection, and state prediction experiments in the [paper](https://arxiv.org/abs/2107.08305).

### Amortized Clustering
### Classification
### Set Anomaly Detection
### State Prediction
