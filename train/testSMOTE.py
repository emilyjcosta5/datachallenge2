import h5py
import numpy as np
import os
import pandas as pd
#import tensorflow as tf
#import hdf5_to_tfrecord
#import pyUSID as usid
#from numba import cuda
import matplotlib.pyplot as plt
import seaborn as sns
import json
from imblearn.over_sampling import SMOTE
import random


path=os.path.join(os.getcwd() , 'batch_train_223.h5')
file = h5py.File(path, 'r')
keys = file.keys()
samples = [file[key] for key in keys]

images=[]
classes=[]

fig, axes = plt.subplots(2,3, figsize=(24, 20))
for ax, cbed in zip(axes.flatten()[:3], file["sample_2_1"]['cbed_stack']):
    ax.imshow(cbed**0.25)
plt.show()


for sample in samples[:10]:
    images.append(sample['cbed_stack'][()].reshape(-1,1))
    classes.append(sample.attrs['space_group'].decode('UTF-8'))

for sample in samples[10:20]:
    images.append(sample['cbed_stack'][()].reshape(-1,1))
    classes.append(sample.attrs['space_group'].decode('UTF-8'))
print(classes)

fig, axes = plt.subplots(2,3, figsize=(24, 20))
for ax, cbed in zip(axes.flatten()[:3], samples[10]['cbed_stack']):
    ax.imshow(cbed**0.25)
title = "Space Group: {} - Original".format(samples[10].attrs['space_group'].decode('UTF-8'))
fig.suptitle(title, size=40)
for ax, cbed in zip(axes.flatten()[3:], samples[11]['cbed_stack']):
    ax.imshow(cbed**0.25)
plt.savefig('original.png')

images=np.squeeze(np.array(images))
sm = SMOTE(random_state=42, k_neighbors=6, ratio={'123':10, '2':15})
images_res, classes_res = sm.fit_resample(images, classes)

images_final=[]

print(classes_res)
image_rest_list=images_res.tolist()
for image_rest_list in image_rest_list:
    images_final.append(np.reshape(image_rest_list, (3, 512, 512)))

print("length of images: {}".format(len(images)))
print("length of images_final: {}".format(len(images_final)))
listNum = random.sample(range(10,24), 4)
print(listNum)
fig, axes = plt.subplots(4, 3, figsize=(24, 20))
for ax, cbed in zip(axes.flatten()[:3], images_final[listNum[0]]):
    ax.imshow(cbed**0.25)
for ax, cbed in zip(axes.flatten()[3:], images_final[listNum[0]]):
        ax.imshow(cbed**0.25)
for ax, cbed in zip(axes.flatten()[6:], images_final[listNum[0]]):
    ax.imshow(cbed**0.25)
for ax, cbed in zip(axes.flatten()[9:], images_final[listNum[0]]):
    ax.imshow(cbed**0.25)
title = "Space Group: {} - Generated".format(classes_res[listNum[0]])
fig.suptitle(title, size=40)
plt.savefig('generated.png')


# print("Original data of class{}: {}".format(classes[-1], samples[-1]['cbed_stack'][()]))
# print("Generated data of class{}: {}".format(classes_res[-1], images_final[-1]))

