# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 15:45:01 2020

@author: Stefan
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.keras.layers import Activation, Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

from tensorflow.keras.utils import plot_model

import numpy as np
import matplotlib.pyplot as plt

import os


os.environ["CUDA_VISIBLE_DEVICES"]="1" # second gpu

E1_input = np.loadtxt("E1.txt", dtype='i', delimiter='\t')
V1_input = np.loadtxt("V1.txt", dtype='i', delimiter='\t')
test_image=np.reshape(E1_input[:,0],(50,50))
print(test_image.shape)
print(type(E1_input))

E1=np.ravel(E1_input,order="F")
E1=np.reshape(E1,(12000,50,50,1))
V1=np.ravel(V1_input,order="F")
V1=np.reshape(V1,(12000,50,50,1))

x_train=E1[0:11000,:,:,:].astype('float32')
x_test=E1[11000:,:,:,:].astype('float32')
y_train=V1[0:11000,:,:,:].astype('float32')
y_test=V1[11000:,:,:,:].astype('float32')

fig=plt.figure(figsize=(8, 4))
for i in range(0,4):
    fig.add_subplot(2, 4, i+1)   # subplot one
    test_image=x_train[i,:,:,0]
    plt.imshow(test_image, cmap=plt.cm.gray)
    
for i in range(5,9):
    fig.add_subplot(2, 4, i)   # subplot one
    test_image=y_train[i-5,:,:,0]
    #test_image=np.reshape(V1_input[:,i-5],(50,50))
    plt.imshow(test_image, cmap=plt.cm.gray)

plt.show()

image_size=E1.shape[1]
# implement NN keras
# Network parameters
input_shape = (image_size, image_size, 1)
batch_size = 128
kernel_size = 3
latent_dim = 20
# Encoder/Decoder number of CNN layers and filters per layer
layer_filters = [128, 256]

# Build the Autoencoder Model
# First build the Encoder Model
inputs = Input(shape=input_shape, name='encoder_input')
x = inputs
# Stack of Conv2D blocks
# Notes:
# 1) Use Batch Normalization before ReLU on deep networks
# 2) Use MaxPooling2D as alternative to strides>1
# - faster but not as good as strides>1
for filters in layer_filters:
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=1,
               activation='relu',
               padding='same')(x)

# Shape info needed to build Decoder Model
shape = K.int_shape(x)

# Generate the latent vector
x = Flatten()(x)
latent = Dense(latent_dim, name='latent_vector')(x)

# Instantiate Encoder Model
encoder = Model(inputs, latent, name='encoder')
encoder.summary()

# Build the Decoder Model
latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
x = Dense(shape[1] * shape[2] * shape[3])(latent_inputs)
x = Reshape((shape[1], shape[2], shape[3]))(x)

# Stack of Transposed Conv2D blocks
# Notes:
# 1) Use Batch Normalization before ReLU on deep networks
# 2) Use UpSampling2D as alternative to strides>1
# - faster but not as good as strides>1
for filters in layer_filters[::-1]:
    x = Conv2DTranspose(filters=filters,
                        kernel_size=kernel_size,
                        strides=1,
                        activation='relu',
                        padding='same')(x)

x = Conv2DTranspose(filters=1,
                    kernel_size=kernel_size,
                    padding='same')(x)

outputs = Activation('sigmoid', name='decoder_output')(x)

# Instantiate Decoder Model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()

# Autoencoder = Encoder + Decoder
# Instantiate Autoencoder Model
autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
autoencoder.summary()

autoencoder.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
plot_model(autoencoder, to_file='model.png',show_shapes=True, show_layer_names=True,rankdir="TB",expand_nested=False,dpi=96,)

# Train the autoencoder
autoencoder.fit(y_train,
                x_train,
                validation_data=(y_test, x_test),
                epochs=5,#35
                batch_size=batch_size)

# Predict the Autoencoder output from corrupted test images
x_decoded = autoencoder.predict(y_test)

fig=plt.figure(figsize=(8, 4))
for i in range(0,4):
    fig.add_subplot(3, 4, i+1)   # subplot one
    test_image=y_test[i,:,:,0]
    plt.imshow(test_image, cmap=plt.cm.gray)
    
for i in range(5,9):
    fig.add_subplot(3, 4, i)   # subplot one
    test_image=x_decoded[i-5,:,:,0]
    #test_image=np.reshape(V1_input[:,i-5],(50,50))
    plt.imshow(test_image, cmap=plt.cm.gray)
for i in range(9,13):
    fig.add_subplot(3, 4, i)   # subplot one
    test_image=x_test[i-9,:,:,0]
    #test_image=np.reshape(V1_input[:,i-5],(50,50))
    plt.imshow(test_image, cmap=plt.cm.gray)

plt.show()

#autoencoder.save("model_ok_50.h5")