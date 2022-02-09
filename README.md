# Neural-Style-Transfer
The Neural Style Transfer algorithm created by [Gatys et al. (2015)](https://arxiv.org/abs/1508.06576).
For most of algorithm such face recognition we optimize a cost function to get a set of parameter values. With Neural Style Transfer, we'll get to optimize a cost function to get pixel values.

## 1- Problem Statment

Neural Style Transfer (NST) is one of the most fun and interesting optimization techniques in deep learning. It merges two images, namely: a "content" image (C) and a "style" image (S), to create a "generated" image (G). The generated image G combines the "content" of the image C with the "style" of image S.

In this code, we are going to combine the Louvre museum in Paris (content image C) with the impressionist style of Claude Monet (content image S) to generate the following image:

<p align="center">
  <img width="800" src="https://github.com/ShafieCoder/Neural-Style-Transfer/blob/main/images/louvre_generated.png" alt="NST-louver">
</p>

Let's get started!

## 2- Transfer Learning

Neural Style Transfer (NST) uses a previously trained convolutional network, and builds on top of that. The idea of using a network trained on a different task and applying it to a new task is called transfer learning.

We will be using the the epynomously named VGG network from the [original NST paper](https://arxiv.org/abs/1508.06576) published by the Visual Geometry Group at University of Oxford in 2014. Specifically, we'll use VGG-19, a 19-layer version of the VGG network. This model has already been trained on the very large ImageNet database, and has learned to recognize a variety of low level features (at the shallower layers) and high level features (at the deeper layers).

## 3- Neural Style Transfer (NST)
Next, we will be building the Neural Style Transfer (NST) algorithm in three steps:
