# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 22:52:07 2020

@author: Stefan
"""

import imageio
import glob
import matplotlib.pyplot as plt
from matplotlib.image import imsave

from keras.models import load_model
import numpy as np

input_tensor=np.zeros((100,50,50,1))
i=0
for im_path in glob.glob("C:\\Users\\Stefan\\Desktop\\image_correction\\Human_performance_test\\Y\\*.png"):
    print(im_path)
    im = imageio.imread(im_path)
    data=im[:,:,1]
    data=np.where(data>1,1,0)
    input_tensor[i,:,:,0]=data
    i=i+1     
     
autoencoder=load_model("model_ok_50.h5")
x_decoded = autoencoder.predict(input_tensor)
i=0
for im_path in glob.glob("C:\\Users\\Stefan\\Desktop\\image_correction\\Human_performance_test\\Y\\*.png"):
    image_file_name=im_path[0:-4] + "_ai.png"
    imsave(image_file_name,x_decoded[i,:,:,0])
    i=i+1


fig=plt.figure(figsize=(8, 4))
fig.add_subplot(1, 1, 1)   # subplot one
plt.imshow(input_tensor[10,:,:,0], cmap=plt.cm.gray)
plt.show()