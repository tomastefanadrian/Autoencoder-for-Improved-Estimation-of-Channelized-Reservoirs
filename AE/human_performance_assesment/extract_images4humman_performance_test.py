# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 12:20:12 2020

@author: Stefan
"""
import numpy as np
import matplotlib

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

idx=np.random.randint(0,high=999,size=100)

human_performance_X=x_test[idx,:,:,:]
human_performance_Y=y_test[idx,:,:,:]

for i in range(0,100):
    image_file_name_X='X_' + str(i)+'.png'
    matplotlib.image.imsave(image_file_name_X,human_performance_X[i,:,:,0]) 
    image_file_name_Y='Y_' + str(i)+'.png'
    matplotlib.image.imsave(image_file_name_Y,human_performance_Y[i,:,:,0]) 