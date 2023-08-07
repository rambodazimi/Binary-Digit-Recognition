# Binary-Digit-Recognition
A Neural Network for Handwritten Digit Recognition, Binary

# Handwritten Digit Recognition Neural Network

![Author: Rambod Azimi](https://img.shields.io/badge/Author-Rambod%20Azimi-blue)
![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x%2B-orange)

This repository contains a Python implementation of a neural network designed to recognize handwritten digits zero and one using TensorFlow and Keras. The neural network architecture includes two hidden layers with sigmoid activation functions and an output layer for binary classification.

## Table of Contents

- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Usage](#usage)
- [Neural Network Architecture](#neural-network-architecture)
- [Training](#training)
- [Testing and Prediction](#testing-and-prediction)

## Getting Started

### Installation
Make sure that you have already installed the required libraries and frameworks on you computer:

TensorFlow

NumPy

os

### Usage
Ensure you have your training data in the appropriate format.

Update the load_data() function in the autils.py file to load your training data.

Run the main script to construct, train, and test the neural network:

bash
Copy code
python main.py

### Neural Network Architecture
The neural network architecture consists of:

Input Layer: 400 units (flattened 20x20 image)
Hidden Layer 1: 25 units, sigmoid activation
Hidden Layer 2: 15 units, sigmoid activation
Output Layer: 1 unit, sigmoid activation (binary classification)

### Training
The model is trained using the Adam optimizer with a learning rate of 0.001 and the Binary Cross-Entropy loss function.

To train the model, run the script main.py. The training process includes 20 epochs.

### Testing and Prediction
The model's predictions can be tested using specific examples from the training dataset. The script main.py demonstrates how to predict and test two examples.
