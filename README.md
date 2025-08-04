# ML-for-High-z-Classification
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
- Residual architecture to capture both spectral and spatial features
- Channel attention module (Squeeze-and-Excitation) to highlight informative bands
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
- Loss function: CrossEntropyLoss with optional class weighting
- Optimiser: Adam, default learning rate 5e-4
- Input: 7-band image cubes (shape: [batch_size, 7, 64, 64])
- Output: 3-class softmax logits

## References

