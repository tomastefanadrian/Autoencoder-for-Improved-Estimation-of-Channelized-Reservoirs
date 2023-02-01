# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 12:54:42 2020

@author: Stefan
"""



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imsave
from keras.models import load_model

E1_input = np.loadtxt("E1.txt", dtype='i', delimiter='\t')
V1_input = np.loadtxt("V1.txt", dtype='i', delimiter='\t')
test_image=np.reshape(E1_input[:,0],(50,50))
print(test_image.shape)
print(type(E1_input))

E1=np.ravel(E1_input,order="F")
E1=np.reshape(E1,(12000,50,50,1))
V1=np.ravel(V1_input,order="F")
V1=np.reshape(V1,(12000,50,50,1))

x_train=E1[0:11000,:,:,:]
x_test=E1[11000:,:,:,:]
y_train=V1[0:11000,:,:,:]
y_test=V1[11000:,:,:,:]


autoencoder=load_model("model_ok_50.h5")
autoencoder.summary()

x_decoded = autoencoder.predict(y_test)

for i in range(0,1000):
    image_file_name_X='X_full_' + str(i)+'.png'
    imsave(image_file_name_X,x_decoded[i,:,:,0]) 
    image_file_name_Y='Y_full_' + str(i)+'.png'
    imsave(image_file_name_Y,y_test[i,:,:,0]) 
    
    
for i in range(0,1000):
    image_file_name_O='O_full_' + str(i)+'.png'
    imsave(image_file_name_O,x_test[i,:,:,0]) 
#fig=plt.figure(figsize=(8, 4))
#for i in range(0,4):
#    fig.add_subplot(3, 4, i+1)   # subplot one
#    test_image=y_test[i,:,:,0]
#    plt.imshow(test_image, cmap=plt.cm.gray)
#    
#for i in range(5,9):
#    fig.add_subplot(3, 4, i)   # subplot one
#    test_image=x_decoded[i-5,:,:,0]
#    #test_image=np.reshape(V1_input[:,i-5],(50,50))
#    plt.imshow(test_image, cmap=plt.cm.gray)
#for i in range(9,13):
#    fig.add_subplot(3, 4, i)   # subplot one
#    test_image=x_test[i-9,:,:,0]
#    #test_image=np.reshape(V1_input[:,i-5],(50,50))
#    plt.imshow(test_image, cmap=plt.cm.gray)
#
#plt.show()