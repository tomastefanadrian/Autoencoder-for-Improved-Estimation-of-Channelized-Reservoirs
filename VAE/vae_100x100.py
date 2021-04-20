# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 23:14:40 2020

@author: Stefan
"""

import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape, Activation
from tensorflow.keras.layers import BatchNormalization
#from tensorflow.keras.layers import MaxPooling2D
#from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

#load dataset
#data=np.load("/home/stefan/image_correction/large_image_train/full_image_data.npz")
data=np.load("/home/stefan/image_correction/vaes/Data/full_image_data.npz")

x_train=data['x_train']
x_test=data['x_test']
y_train=data['y_train']
y_test=data['y_test']

x_full=np.concatenate((x_train,x_test))

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

image_size=x_train.shape[1]
input_shape = (image_size, image_size, 1)
batch_size = 32
#kernel_size = 3
latent_dim = 100
img_width, img_height = x_train.shape[1], x_train.shape[2]
no_epochs = 5
validation_split = 0.2

# Normalize data
x_train = x_train / 255
x_test = x_test / 255

# # =================
# # Encoder
# # =================

# Definition
i       = Input(shape=input_shape, name='encoder_input')
cx      = Conv2D(filters=16, kernel_size=3, strides=1, padding='same', activation='relu')(i)
cx      = MaxPooling2D(pool_size=(2,2),strides=None,padding='same')(cx)
cx      = Conv2D(filters=32, kernel_size=3, strides=2, padding='same', activation='relu')(cx)
cx      = Conv2D(filters=32, kernel_size=2, strides=1, padding='same', activation='relu')(cx)
cx      = Conv2D(filters=64, kernel_size=2, strides=2, padding='same', activation='relu')(cx)
cx      = MaxPooling2D(pool_size=(2,2),strides=None,padding='valid')(cx)
x       = Flatten()(cx)
x       =  Dense(2000, activation='relu')(x)
x       = BatchNormalization()(x)
x       = Activation(activation='relu')(x)
mu      = Dense(latent_dim, activation='linear', name='latent_mu')(x)
sigma   = Dense(latent_dim, activation='linear', name='latent_sigma')(x)

# Get Conv2D shape for Conv2DTranspose operation in decoder
conv_shape = K.int_shape(cx)

# Define sampling with reparameterization trick
def sample_z(args):
  mu, sigma = args
  batch     = K.shape(mu)[0]
  dim       = K.int_shape(mu)[1]
  eps       = K.random_normal(shape=(batch, dim))
  return mu + K.exp(sigma / 2) * eps

# Use reparameterization trick to ensure correct gradient
z       = Lambda(sample_z, output_shape=(latent_dim, ), name='z')([mu, sigma])

# Instantiate encoder
encoder = Model(i, [mu, sigma, z], name='encoder')
encoder.summary()


# =================
# Decoder
# =================

# Definition
d_i   = Input(shape=(latent_dim, ), name='decoder_input')
x     = Dense(2000, activation='relu')(d_i)
x     = BatchNormalization()(x)
x     = Activation(activation='relu')(x)
x     = Dense(20000, activation='relu')(x)
x     = Reshape((25,25,32))(x)
cx      = Conv2DTranspose(filters=64, kernel_size=2, strides=2, padding='same', activation='relu')(x)
cx      = Conv2DTranspose(filters=32, kernel_size=2, strides=1, padding='same', activation='relu')(cx)
cx      = Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu')(cx)
cx      = Conv2DTranspose(filters=16, kernel_size=3, strides=1, padding='same', activation='relu')(cx)
o    = Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same', activation='sigmoid')(cx)


# Instantiate decoder
decoder = Model(d_i, o, name='decoder')
decoder.summary()

# =================
# VAE as a whole
# =================

# Instantiate VAE
vae_outputs = decoder(encoder(i)[2])
vae         = Model(i, vae_outputs, name='vae')
vae.summary()

# Define loss
def kl_reconstruction_loss(true, pred):
  # Reconstruction loss
  reconstruction_loss = binary_crossentropy(K.flatten(true), K.flatten(pred)) * img_width * img_height
  # KL divergence loss
  kl_loss = 1 + sigma - K.square(mu) - K.exp(sigma)
  kl_loss = K.sum(kl_loss, axis=-1)
  kl_loss *= -0.5
  # Total loss = 50% rec + 50% KL divergence loss
  return K.mean(reconstruction_loss + kl_loss)

# Compile VAE
vae.compile(optimizer='adam', loss=kl_reconstruction_loss,experimental_run_tf_function=False)

# Train autoencoder
vae.fit(x_full, x_full, epochs = no_epochs, batch_size = batch_size, validation_split = validation_split)


def viz_decoded(decoder):
    fig, axs = plt.subplots(3, 3)
    #num_samples = 9
    mean = np.zeros((1, 100))
    mean = np.reshape(mean, -1)
    cov = np.identity(100)
    num_channels = 1
    for i in range(3):
        for j in range(3):
            decoderInput = np.random.default_rng().multivariate_normal(mean, cov)
            decoderInput = decoderInput.reshape((1, 100))
            #print(type(decoderInput))
            #print(decoderInput.shape)
            x_decoded = decoder.predict(decoderInput)
            digit = x_decoded[0].reshape(img_width, img_height, num_channels)
            genNumber = digit[:, :, 0]
            axs[i, j].imshow(genNumber, cmap=plt.cm.gray)


viz_decoded(decoder)



# # =================
# # Results visualization
# # Credits for original visualization code: https://keras.io/examples/variational_autoencoder_deconv/
# # (François Chollet).
# # Adapted to accomodate this VAE.
# # =================






# # def viz_latent_space(encoder, data):
# #   input_data, target_data = data
# #   mu, _, _ = encoder.predict(input_data)
# #   plt.figure(figsize=(8, 10))
# #   plt.scatter(mu[:, 0], mu[:, 1], c=target_data)
# #   plt.xlabel('z - dim 1')
# #   plt.ylabel('z - dim 2')
# #   plt.colorbar()
# #   plt.show()
  
