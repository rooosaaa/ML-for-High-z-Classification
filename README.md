# Machine Learning Techniques for High-Redshift Galaxy Classification with JWST NIRCam Data
## Author
Rosa Roberts (rosa.roberts@student.manchester.ac.uk)
- Jodrell Bank Centre for Astrophysics, University of Manchester, Manchester, M13 9PL, UK.

## Date
04.08.2025

## Overview
This repository contains a custom Convolutional Neural Network (CNN) built with PyTorch to classify galaxies from JWST imaging into one of three categories:
- High redshift (z ≥ 4)
- Low redshift (z < 4)
- Brown dwarf contaminants
  
The model is tailored for 7-band JWST NIRCam cutouts, using both spectral and spatial information to distinguish between distant galaxies and stellar interlopers. It’s designed for astrophysical surveys aiming to identify candidate galaxies from the early Universe.

## Features
- Residual architecture to capture both spectral and spatial features [1]
- Channel attention module (Squeeze-and-Excitation) to highlight informative bands [2]
- Modular design for easy modification or experimentation
- Includes tools to extract intermediate feature maps for visualisation or interpretation
- Optimised for 64×64 pixel image cutouts across 7 JWST NIRCam filters

## Model Overview
The model consists of:
- SpectralResidualBlock: Captures correlations across bands with 1×1 convolutions
- SpatialResidualBlock: Learns spatial patterns (e.g. morphology) using 3×3 convolutions
- ChannelAttention: Dynamically reweights spectral channels to focus on informative filters
- Classifier: Fully connected layers with batch normalisation and dropout for final prediction

## Getting Started
To instantiate the model:
```bash
from model import create_model
model = create_model(n_bands=7, image_size=64)
```
To train, pass the model, dataloaders, and optimiser into your training loop.

## Training Details
- Loss function: CrossEntropyLoss with optional class weighting [3]
- Optimiser: Adam, default learning rate 5e-4 [4]
- Input: 7-band image cubes (shape: [batch_size, 7, 64, 64])
- Output: 3-class softmax logits [5]

## Acknowledgements

This repository was developed as part of a summer research project focused on machine learning methods for identifying high-redshift galaxies in JWST NIRCam imaging. The study explores both supervised CNN models and a semi-supervised pipeline combining HDBSCAN clustering with Random Forest classification. The supervised pipeline aims to detect key spectral features like the Lyman break, achieving up to 98% accuracy in classifying galaxies by redshift. Funding for this project was provided by Google DeepMind, the Royal Academy of Engineering, and the Hg Foundation. 

## References

[1] Kaiming He, Xiangyu Zhang, et al. Deep Residual Learning for Image Recognition, IEEE, 2015.

[2] Jie Hu, Li Shen, et al. Squeeze-and-Excitation Networks, IEEE, 2019.

[3] Sara A. Solla, Esther Levin, et al, Accelerated learning in layered neural networks, Complex Systems, 2:625–640, 1988.

[4] Diederik P. Kingma and Jimmy Ba, Adam: A method for stochastic optimization, In International Conference on Learning Representations, 2017.

[5] John S. Bridle, Training Stochastic Model Recognition Algorithms as Networks can Lead to Maximum Mutual Information Estimation of Parameters, Advances in Neural Information Processing Systems 2, 1989.





