""""
Created on: 7.8.19
Author: Emily Costa, Shuto Araki
This program processes data prior to running through ML algorithm.
My current idea is to do some numeric approach to decide which data is worth keeping
"""

import h5py
import numpy as np
import os
import matplotlib.pyplot as plt

h5_path = os.getcwd() + "/train"

def display_image(key):
    filename = os.path.join(h5_path, "batch_train_223.h5")
    f = h5py.File(filename, 'r')
    sample = f[key]
    # print(list(sample.attrs.items()))
    # This is the image dataset (array/file)
    cbed_stack = sample['cbed_stack']

    # Display the images
    fig, axes = plt.subplots(1, 3, figsize=(16, 12))
    for ax, cbed in zip(axes.flatten(), cbed_stack):
        ax.imshow(cbed**0.25)
    
    title = "Index: {}\nSpace Group: {}".format(key, sample.attrs['space_group'])
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    key = 'sample_0_0'
    display_image(key)
