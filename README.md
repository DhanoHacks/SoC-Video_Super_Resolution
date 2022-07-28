# Final Submission
This is __the Final Submission__ of Seasons Of Code 2022 - Video Super Resolution.

## Problem Statement
Consider a single low-resolution image, we first upscale it to the desired size using bicubic interpolation to obtain Y. Our goal is to recover from Y, an image F(Y) that is as similar as possible to the actual high resolution image X.
The Image Super Resolution Problem has been taken further, into what we can call "Naive Video Super Resolution", where each frame of the low resolution video is processed as above to obtain it in high resolution, and all these frames are put back together to create the higher resolution video.

## Step 1 - Preparing Data
I have used the T91 data (in HDF5 binary file format) to train my models (drive link- https://drive.google.com/drive/folders/1L3CF-uAHILPIlQfGYmsftvVHNXQceVVC?usp=sharing ).

## Step 2 - Bicubic Interpolation
I performed bicubic interpolation on the images to obtain poor-quality upscaling. This will act as a basis of input for the neural network.

## Step 3 - Training the Model
I created a neural network model with three convolutional layers, and a ReLu activation layer between each convolutional layer.
As part of preprocessing, the input image is padded on all sides to obtain the output with same dimensions as input.
The goal of this step is to train the neural network on the above data. I have saved multiple versions of the model parameters inside the "models" folder, along with the hyperparameters used for each. Two versions in each scale (x2, x3, x4). Ver 1 was my first attempt at training, Ver 2 and Ver 3 are better.

I have included the command to run my model in run.txt

## Step 4 - Testing the Model
The goal of this step is to test the model on sample data.
I calculated the PSNR (Peak Signal to Noise Ratio, metric used to evaluate how close the super resolved image is to the original high resolution image) for each model (and each scale), and documented it in the "parameters.txt" file for each version.
Running "viewer.py" also opens the three versions of image (original low res version, upscaled bicubic version, and super resolved version) side-by-side in matplotlib, so that the images can be compared manually.

## Conclusion
Drive link for output files: https://drive.google.com/drive/folders/1l8Z9KW4FaVbX0O_ltOSXmaPG0B0JQhfS?usp=sharing
At the end of this project I have successfully implemented an SRCNN model discussed in a Research Paper, and tweaking it by changing the hyperparameters to make it even better.