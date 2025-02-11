# CNN-from-scratch
# Convolutional Neural Network Implementation

This project provides a step-by-step implementation of key components in Convolutional Neural Networks (CNNs), including:

- **Convolution**  
- **Max Pooling**  
- **Batch Normalization**

The code is designed to demonstrate both the forward and backward pass implementations for these operations.

## Files Overview

- **convolutional_networks.py**  
  This file contains the implementation of the forward and backward passes for the convolution, max-pooling, and batch normalization layers.

- **convolutional_networks.ipynb**  
  This Jupyter notebook contains the results and testing of the CNN model using the implementations from `convolutional_networks.py`.

## Features

1. **Convolution Layer**: 
   - Performs the convolution operation with different filter sizes and strides.
   - Includes the forward pass for generating feature maps and the backward pass for updating the weights and gradients.

2. **Max Pooling Layer**: 
   - Implements max pooling with custom kernel size and stride.
   - The forward pass extracts the most significant features, and the backward pass updates the gradients.

3. **Batch Normalization**: 
   - Normalizes the input of each layer to improve training stability and convergence.
   - Both forward and backward passes are implemented.
