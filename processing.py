""""
Created on: 7.8.19
Author: Emily Costa, Shuto Araki
This program processes data prior to running through ML algorithm.
My current idea is to do some numeric approach to decide which data is worth keeping
"""

import h5py
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import hdf5_to_tfrecord
import pyUSID as usid


def display_image(f, key):
    """Display 3 images for a given sample index.
    Args:
        f: h5py.File object
        key: string of sample index (e.g., 'sample_0_0')
    """
    import matplotlib.pyplot as plt
    sample = f[key]
    # This is the image dataset (array/file)
    cbed_stack = sample['cbed_stack']

    # Display the images
    fig, axes = plt.subplots(1, 3, figsize=(16, 12))
    for ax, cbed in zip(axes.flatten(), cbed_stack):
        ax.imshow(cbed**0.25)

    title = "Index: {}\nSpace Group: {}".format(key, sample.attrs['space_group'])
    plt.title(title)
    plt.savefig('cbed_stack.png')

def display_space_group_dist(f):
    """Display a histogram of space group distribution of 
    a given h5py File object
    Args:
        f: h5py.File object
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    keys = list(f.keys())
    samples = [f[key] for key in keys]
    space_groups = [int(sample.attrs['space_group']) for sample in samples]
    space_groups = pd.Series(space_groups, name="Space Groups Distribution")
    sns.distplot(space_groups)
    plt.savefig('histogram_dist.png')

def know_space_groups(f):
    dist = np.zeros(230, dtype=int)
    keys = list(f.keys())
    samples = [f[key] for key in keys]
    space_groups = [int(sample.attrs['space_group']) for sample in samples]
    for space_group in space_groups:
        dist[space_group - 1] += 1

def iterate_through_data():
    for file in files:
        try:
            #open a file and run through know_space_groups

def show_tree(f):
    """
    Displays a tree of the h5.File object
    Groups and datasets
    """
    print("h5 file contains:")
    usid.io.hdf_utils.print_tree(f)


if __name__ == '__main__':
    h5_path = os.getcwd() + "/train"

    filename = os.path.join(h5_path, "batch_train_223.h5")
    f = h5py.File(filename, 'r')

    display_space_group_dist(f)
    show_tree(f)

    f.close()

