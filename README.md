# PyTorch Computer Vision Course

This repository contains code and notes for a comprehensive course on computer vision using PyTorch. The course covers a variety of topics, from basic data handling to building and evaluating complex convolutional neural networks.

## Table of Contents

1. [Computer Vision Libraries in PyTorch](#computer-vision-libraries-in-pytorch)
2. [Load Data](#load-data)
3. [Prepare Data](#prepare-data)
4. [Model 0: Building a Baseline Model](#model-0-building-a-baseline-model)
5. [Making Predictions and Evaluating Model 0](#making-predictions-and-evaluating-model-0)
6. [Setup Device Agnostic Code for Future Models](#setup-device-agnostic-code-for-future-models)
7. [Model 1: Adding Non-Linearity](#model-1-adding-non-linearity)
8. [Model 2: Convolutional Neural Network (CNN)](#model-2-convolutional-neural-network-cnn)
9. [Comparing Our Models](#comparing-our-models)
10. [Evaluating Our Best Model](#evaluating-our-best-model)
11. [Making a Confusion Matrix](#making-a-confusion-matrix)
12. [Saving and Loading the Best Performing Model](#saving-and-loading-the-best-performing-model)

## Computer Vision Libraries in PyTorch

PyTorch offers a variety of built-in libraries that are helpful for computer vision tasks. These libraries make it easier to handle data, build models, and evaluate performance.

## Load Data

To practice computer vision, we start with some images of different pieces of clothing from the FashionMNIST dataset.

## Prepare Data

We load the images using a PyTorch DataLoader, allowing us to use them in our training loop efficiently.

## Model 0: Building a Baseline Model

We create a multi-class classification model to learn patterns in the FashionMNIST data. This section includes choosing a loss function, optimizer, and building a training loop.

## Making Predictions and Evaluating Model 0

We make predictions using our baseline model and evaluate its performance.

## Setup Device Agnostic Code for Future Models

To ensure our code runs on any available device (CPU or GPU), we set up device-agnostic code.

## Model 1: Adding Non-Linearity

Experimenting is a key part of machine learning. Here, we try to improve our baseline model by adding non-linear layers.

## Model 2: Convolutional Neural Network (CNN)

We introduce convolutional neural networks (CNNs), a powerful architecture specific to computer vision tasks.

## Comparing Our Models

After building three different models, we compare their performance to see which one works best.

## Evaluating Our Best Model

We make predictions on random images and evaluate our best model in more detail.

## Making a Confusion Matrix

A confusion matrix is a great way to evaluate a classification model. This section demonstrates how to create one.

## Saving and Loading the Best Performing Model

Since we might want to use our model later, we save it and ensure it loads back in correctly.

## Installation

To run the code in this repository, you'll need to have Python and the following packages installed:

- PyTorch
- Torchvision
- Torchmetrics
- Matplotlib
- TQDM
- Mlxtend

You can install the required packages using pip:

Acknowledgements
Special thanks to Daniel Bourke (mrdbourke) for his excellent tutorials and guidance on learning PyTorch and computer vision. His resources have been invaluable in creating this repository. 
You can find more of his work on GitHub and his website.

```bash
pip install torch torchvision torchmetrics matplotlib tqdm mlxtend
