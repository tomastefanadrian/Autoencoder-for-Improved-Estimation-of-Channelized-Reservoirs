# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#import keras
from tensorflow.keras.layers import Activation, Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import Reshape, Conv2DTranspose, MaxPooling2D
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.optimizers import SGD, Nadam, Adam
from tensorflow.keras.callbacks import CSVLogger

from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import timeit
import os

os.environ["CUDA_VISIBLE_DEVICES"]="1" # second gpu

data=np.load("full_image_data.npz")

x_train=data['x_train'].astype('float32')
x_test=data['x_test'].astype('float32')
y_train=data['y_train'].astype('float32')
y_test=data['y_test'].astype('float32')
x_dev=x_test[0:1500,:,:,:]
y_dev=y_test[0:1500,:,:,:]

x_test=x_test[1500:,:,:,:]
y_test=y_test[1500:,:,:,:]

##V7 24000 validation 6000
##only for V4
X_full=np.concatenate((x_train,x_dev,x_test))
Y_full=np.concatenate((y_train,y_dev,y_test))

augmentedX=np.zeros((60000,100,100,1))
augmentedX[:30000,:,:,:]=X_full

augmentedY=np.zeros((60000,100,100,1))
augmentedY[:30000,:,:,:]=Y_full

for i in range(30000):
    augmentedX[30000+i,:,:,0]=np.flip(X_full[i,:,:,0],0)
    augmentedY[30000+i,:,:,0]=np.flip(Y_full[i,:,:,0],0)

newIndex=np.random.permutation(60000)
randAugmentedX=augmentedX[newIndex,:,:,:]
randAugmentedY=augmentedY[newIndex,:,:,:]
   
# x_train=randAugmentedX[0:55000,:,:,:]
# x_dev=randAugmentedX[55000:59000,:,:,:]
# x_test=randAugmentedX[59000:,:,:,:]

# y_train=randAugmentedY[0:55000,:,:,:]
# y_dev=randAugmentedY[55000:59000,:,:,:]
# y_test=randAugmentedY[59000:,:,:,:]


x_train=X_full[0:24000,:,:,:]
x_dev=X_full[24000:30000,:,:,:]
x_test=X_full[24000:,:,:,:]

y_train=Y_full[0:24000,:,:,:]
y_dev=Y_full[24000:30000,:,:,:]
y_test=Y_full[24000:,:,:,:]

image_size = x_train.shape[1]
input_shape = (image_size, image_size, 1)
batch_size = 16
#kernel_size = 15
latent_dim =400
# Encoder/Decoder number of CNN layers and filters per layer
layer_filters = [128, 256]
layer_kernel = [ 3,3]#3 ok
layer_strides = [2,2] 

# Build the Autoencoder Model
# First build the Encoder Model
inputs = Input(shape=input_shape, name='encoder_input')
x = inputs
# Stack of Conv2D blocks
# Notes:
# 1) Use Batch Normalization before ReLU on deep networks
# 2) Use MaxPooling2D as alternative to strides>1
# - faster but not as good as strides>1
for filters,kernel_size,strides in zip(layer_filters,layer_kernel,layer_strides):
    x = Conv2D(filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                activation='relu',
                padding='same')(x)
    x=BatchNormalization()(x)

    
x = Dropout(0.8)(x)
    
# x = Conv2D(filters=layer_filters[0],
#                 kernel_size=layer_kernel[0], 
#                 strides=layer_strides[0],
#                 activation='relu',
#                 padding='same')(x)


# #x=MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding="same")(x)
# x = Dropout(0.5)(x)
# #x=BatchNormalization()(x)
# x = Conv2D(filters=layer_filters[1],
#                 kernel_size=layer_kernel[1], 
#                 strides=layer_strides[1],
#                 activation='relu',
#                 padding='same')(x)

#x=MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding="same")(x)
#x = Dropout(0.8)(x)


#x = Dropout(0.5)(x)


# Shape info needed to build Decoder Model
shape = K.int_shape(x)

# Generate the latent vector
x = Flatten()(x)
latent = Dense(latent_dim, name='latent_vector')(x)
#
## Instantiate Encoder Model
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

x = Dropout(0.8)(x)

for filters,kernel_size,strides in zip(layer_filters[::-1],layer_kernel[::-1],layer_strides[::-1]):
    x = Conv2DTranspose(filters=filters,
                        kernel_size=kernel_size,
                        strides=strides,
                        activation='relu',
                        padding='same')(x)
    x=BatchNormalization()(x)


#tf.keras.layers.MaxPooling2D(
#    pool_size=(2, 2), strides=None, padding="valid", data_format=None, **kwargs
#)
#x = Dropout(0.8)(x)


# x = Conv2DTranspose(filters=layer_filters[1],
#                 kernel_size=layer_kernel[1], 
#                 strides=layer_strides[1],
#                 activation='relu',
#                 padding='same')(x)



# #x=MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding="same")(x)
# x=Dropout(0.5)(x)
# #x=BatchNormalization()(x)


# x = Conv2DTranspose(filters=layer_filters[0],
#                 kernel_size=layer_kernel[0], 
#                 strides=layer_strides[0],
#                 activation='relu',
#                 padding='same')(x)


x = Conv2DTranspose(filters=1,
                    kernel_size=3,
                    padding='same')(x)

outputs = Activation('sigmoid', name='decoder_output')(x)

# Instantiate Decoder Model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()

# Autoencoder = Encoder + Decoder
# Instantiate Autoencoder Model
autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
autoencoder.summary()

# autoencoder=load_model('model_100_100_temp.h5')


# opt = SGD(lr=0.05,momentum=0.9,decay=1e-6)
#opt=Adam()
opt=Nadam(lr=1e-5)
# opt=RMSprop
autoencoder.compile(loss='mse', optimizer=opt,metrics=['accuracy'])
#autoencoder.compile(loss='mse', optimizer=opt,metrics=['accuracy'])

startTrainingTime=timeit.default_timer()
#
csv_logger = CSVLogger('training_cont211.log', separator=',', append=False)
# Train the autoencoder
history = autoencoder.fit(y_train,
                x_train,
                validation_data=(y_dev, x_dev),
                epochs=20,
                batch_size=batch_size,callbacks=[csv_logger])

endTrainingTime=timeit.default_timer()

print("Training time : %f " % (endTrainingTime-startTrainingTime))

print(history.history.keys())
#  "Accuracy"
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# Predict the Autoencoder output from corrupted test images

#autoencoder.save("model_100_100_temp.h5")
##v1 - 5 epoci; train 28000; dev 2000 
##v2 - 20 epoci train 28000 dev 2000 test 2000 - cel mai bun
##v3 - 30 epoci train 28000 dev 1500 test 500 - 0.9782 - pare supra antrenat
##v4 - 40 epoci train 28000 dev 1500 test 500 - 0.9827 - pare supra antrenat
##v5 - 20 epoci train 27000 dev 2500 test 500 - 
##v6 - perfect ?
##v7 the best so far
x_decoded = autoencoder.predict(y_test)

fig=plt.figure(figsize=(8, 4))
for i in range(0,4):
    fig.add_subplot(3, 4, i+1)   # subplot one
    test_image=x_train[i,:,:,0]
    plt.imshow(test_image, cmap=plt.cm.gray)
    
for i in range(5,9):
    fig.add_subplot(3, 4, i)   # subplot one
    test_image=y_train[i-5,:,:,0]
    plt.imshow(test_image, cmap=plt.cm.gray)
plt.show()

fig=plt.figure(figsize=(8, 4))
for i in range(0,4):
    fig.add_subplot(3, 4, i+1)   # subplot one
    test_image=y_test[i,:,:,0]
    plt.imshow(test_image, cmap=plt.cm.gray)
    
for i in range(5,9):
    fig.add_subplot(3, 4, i)   # subplot one
    test_image=x_decoded[i-5,:,:,0]
    plt.imshow(test_image, cmap=plt.cm.gray)

for i in range(9,13):
    fig.add_subplot(3, 4, i)   # subplot one
    test_image=x_test[i-9,:,:,0]
    plt.imshow(test_image, cmap=plt.cm.gray)
plt.show()
