# Neural-Style-Transfer
The Neural Style Transfer algorithm created by [Gatys et al. (2015)](https://arxiv.org/abs/1508.06576).
For most of algorithm such face recognition we optimize a cost function to get a set of parameter values. With Neural Style Transfer, we'll get to optimize a cost function to get pixel values.

## 1- Problem Statment

Neural Style Transfer (NST) is one of the most fun and interesting optimization techniques in deep learning. It merges two images, namely: a "content" image (C) and a "style" image (S), to create a "generated" image (G). The generated image G combines the "content" of the image C with the "style" of image S.

In this code, we are going to combine the Louvre museum in Paris (content image C) with the impressionist style of Claude Monet (content image S) to generate the following image:



