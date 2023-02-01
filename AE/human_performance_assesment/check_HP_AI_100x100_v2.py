# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 22:52:07 2020

@author: Stefan
"""

import matplotlib.pyplot as plt
from matplotlib.image import imsave
from matplotlib.image import imread

from tensorflow.keras.models import load_model
import numpy as np
#from random import shuffle
from scipy.ndimage import gaussian_filter

import os


y_HP=np.zeros((100,100,100,1))
x_HP=np.zeros((100,100,100,1))

a=os.listdir('./HPA')

for file in a:
    y_image=imread('./HPA/'+file)
    y_HP[int(file[2:-4]),:,:,0]=y_image[:,:,1]

a=os.listdir('./orig_HPA')

for file in a:
    x_image=imread('./orig_HPA/'+file)
    x_HP[int(file[2:-4]),:,:,0]=x_image[:,:,1]
    
# fig=plt.figure
# plt.imshow(x_HP[10,:,:,0])
# plt.show()

autoencoder=load_model("model_100_100_v7.h5")

x_decoded = autoencoder.predict(y_HP)

x_processed=np.zeros(x_decoded.shape)

threshold=0.5
for i in range(0,100):
      x_processed[i,:,:,0]=gaussian_filter(x_decoded[i,:,:,0],0.9)
      x_processed[i,:,:,0]=np.where(x_processed[i,:,:,0]>threshold, 255., 0.)

fig=plt.figure(figsize=(8, 4))
for i in range(0,4):
    fig.add_subplot(3, 4, i+1)   # subplot one
    test_image=y_HP[i,:,:,0]
    plt.imshow(test_image, cmap=plt.cm.gray)
    
for i in range(5,9):
    fig.add_subplot(3, 4, i)   # subplot one
    test_image=x_processed[i-5,:,:,0]
    plt.imshow(test_image, cmap=plt.cm.gray)

for i in range(9,13):
    fig.add_subplot(3, 4, i)   # subplot one
    test_image=x_HP[i-9,:,:,0]
    plt.imshow(test_image, cmap=plt.cm.gray)
plt.show()


for i in range(0,100):
    image_file_name_out='./proc_HPA_05/X_proc_'+str(i)+'.png'
    imsave(image_file_name_out,x_processed[i,:,:,0])


# x_out=np.zeros((100,100,100,1))
# for i,j in zip(ind_list,range(0,100)):
#     x_out[j,:,:,0]=np.transpose(x_processed[i,:,:,0])/255
    
    
# HP_processed=np.ravel(x_out,order="F")
# HP_processed=np.reshape(HP_processed,(10000,100))
# y_HP=np.ravel(y_test[ind_list],order="F")
# y_HP=np.reshape(y_HP,(10000,100))
# x_HP=np.ravel(x_test[ind_list],order="F")
# x_HP=np.reshape(x_HP,(10000,100))

# #np.savetxt("E1_1.txt",E1_1,fmt="%d",delimiter="\t")
# #np.savetxt("E1_2.txt",E1_1,fmt="%d",delimiter="\t")
# #np.savetxt("E1_3.txt",E1_1,fmt="%d",delimiter="\t")
# #np.savetxt("E1_4.txt",E1_1,fmt="%d",delimiter="\t")
# np.savetxt("HP_proc.txt",HP_processed,fmt="%d",delimiter="\t")
# np.savetxt("HP_X.txt",x_HP,fmt="%d",delimiter="\t")
# np.savetxt("HP_Y.txt",y_HP,fmt="%d",delimiter="\t")

# for i in ind_list:
#     image_file_name_in='./HPA/Y_'+str(i)+'.png'
#     image_file_name_out='./proc_HPA/X_proc_'+str(i)+'.png'
#     image_file_name_orig='./orig_HPA/X_'+str(i)+'.png'
#     imsave(image_file_name_in,y_test[i,:,:,0])
#     imsave(image_file_name_out,x_processed[i,:,:,0])
#     imsave(image_file_name_orig,x_test[i,:,:,0])


# #