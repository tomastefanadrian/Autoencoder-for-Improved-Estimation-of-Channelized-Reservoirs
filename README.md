# Autoencoder-for-Improved-Estimation-of-Channelized-Reservoirs

This repository hold the implementations of the autoencoders used in "Bridging Deep Convolutional Autoencoders and Ensemble Smoothers for Improved Estimation of Channelized Reservoirs". 

If you use this software (i.e., the autoencoder) please cite the following paper:

Sebacher, B., Toma, S.A. Bridging Deep Convolutional Autoencoders and Ensemble Smoothers for Improved Estimation of Channelized Reservoirs. Math Geosci 54, 903–939 (2022). https://doi.org/10.1007/s11004-022-09997-7

If you use the VAE please cite this repository.

Folders:

Data - input data (training and testing) - not pushed to git, because the files are too large  
Models - NN models; not pushed, too large  
  
AE - autoencoders for 50 x 50 and 100 x 100 images  
VAE - variational autoencoders for 50 x 50 and 100 x 100 images  

All images have 2 types of rocks: background and channel; binary images (1 and 0)  
For example, a training instance can be 50 x 50 x 1  

![alt text](https://github.com/tomastefanadrian/Autoencoder-for-Improved-Estimation-of-Channelized-Reservoirs/blob/main/AE/examples.png)

=======
