# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras.models import load_model
import numpy as np
from scipy.ndimage import gaussian_filter

#import keras
#from keras.layers import Activation, Dense, Input
#from keras.layers import Conv2D, Flatten
#from keras.layers import Reshape, Conv2DTranspose
#from keras.models import Model
#from keras import backend as K
#from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
#from matplotlib.image import imsave

#V1_real_input = np.loadtxt("V1_1.txt", dtype='i', delimiter='\t')
#V1_real_input = np.loadtxt("V1_2.txt", dtype='i', delimiter='\t')
#V1_real_input = np.loadtxt("V1_3.txt", dtype='i', delimiter='\t')
#V1_real_input = np.loadtxt("V1_4.txt", dtype='i', delimiter='\t')

V1_real_input = np.loadtxt(".\\model_complex\\V1_4.txt", dtype='i', delimiter='\t')



V1=np.ravel(V1_real_input,order="F")
V1=np.reshape(V1,(120,50,50,1))
input_tensor=V1[0:,:,:,:]

autoencoder=load_model("C:\\Users\\Stefan\\Desktop\\image_correction\\model_ok_50.h5")
x_decoded = autoencoder.predict(input_tensor)

x_processed=np.zeros(x_decoded.shape)

threshold=0.3
for i in range(0,120):
    x_processed[i,:,:,0]=gaussian_filter(x_decoded[i,:,:,0],0.9)
    x_processed[i,:,:,0]=np.where(x_processed[i,:,:,0]>threshold, 255., 0.)

fig=plt.figure(figsize=(8, 4))
for i in range(0,4):
    fig.add_subplot(3, 4, i+1)   # subplot one
    test_image=x_decoded[i,:,:,0]
    plt.imshow(test_image, cmap=plt.cm.gray)
    
for i in range(5,9):
    fig.add_subplot(3, 4, i)   # subplot one
    test_image=input_tensor[i-5,:,:,0]
    plt.imshow(test_image, cmap=plt.cm.gray)
    
for i in range(9,13):
    fig.add_subplot(3, 4, i)   # subplot one
    test_image=x_processed[i-9,:,:,0]
    plt.imshow(test_image, cmap=plt.cm.gray)
plt.show()
for i in range(0,120):
    x_processed[i,:,:,0]=np.transpose(x_processed[i,:,:,0])/255
E1_1=np.ravel(x_processed,order="F")
E1_1=np.reshape(E1_1,(2500,120))
#np.savetxt("E1_1.txt",E1_1,fmt="%d",delimiter="\t")
#np.savetxt("E1_2.txt",E1_1,fmt="%d",delimiter="\t")
#np.savetxt("E1_3.txt",E1_1,fmt="%d",delimiter="\t")
#np.savetxt("E1_4.txt",E1_1,fmt="%d",delimiter="\t")
np.savetxt(".\\model_complex\\E1_4.txt",E1_1,fmt="%d",delimiter="\t")


#
#for i in range(0,120):
#    image_file_name_in='C:\\Users\\Stefan\\Desktop\\image_correction\\Real_world_results\\V1_in_'+str(i)+'.png'
#    image_file_name_out='C:\\Users\\Stefan\\Desktop\\image_correction\\Real_world_results\\V1_out_'+str(i)+'.png'
#    imsave(image_file_name_in,input_tensor[i,:,:,0])
#    imsave(image_file_name_out,x_processed[i,:,:,0])



