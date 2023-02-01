# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 17:53:44 2020

@author: Stefan
"""

import numpy as np
import matplotlib.pyplot as plt

E1_input = np.loadtxt("E1.txt", dtype='i', delimiter='\t')
V1_input = np.loadtxt("V1.txt", dtype='i', delimiter='\t')
test_image=np.reshape(E1_input[:,0],(50,50))
print(test_image.shape)
print(type(E1_input))

E1=np.ravel(E1_input,order="F")
E1=np.reshape(E1,(12000,50,50,1))
V1=np.ravel(V1_input,order="F")
V1=np.reshape(V1,(12000,50,50,1))

x_train=E1[0:10000,:,:,:]
x_test=E1[10000:,:,:,:]
y_train=V1[0:10000,:,:,:]
y_test=V1[10000:,:,:,:]
i=1;

for j in range(0,100):
    if (i==1):
        fig=plt.figure()
    fig.add_subplot(5, 5, i)   # subplot one
    test_image=x_train[j,:,:,0]
    plt.imshow(test_image, cmap=plt.cm.gray)
    i=i+1
    if (i==26):
        plt.show()
        plt.waitforbuttonpress()
        plt.close(fig)
        i=1
        
#fig=plt.figure()
#fig.add_subplot(5, 5, 1)   # subplot one
#test_image=y_train[0,:,:,0]
#plt.imshow(test_image, cmap=plt.cm.gray)
#fig.add_subplot(5, 5, 2)   # subplot one
#test_image=y_train[1,:,:,0]
#plt.imshow(test_image, cmap=plt.cm.gray)
#plt.show()
#plt.waitforbuttonpress()
#plt.close(fig)
