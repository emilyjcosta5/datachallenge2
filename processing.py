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
#import pyUSID as usid
#from numba import cuda
import matplotlib.pyplot as plt
import seaborn as sns
import json

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
    keys = list(f.keys())
    samples = [f[key] for key in keys]
    space_groups = [int(sample.attrs['space_group']) for sample in samples]
    space_groups = pd.Series(space_groups, name="Space Groups Distribution")
    sns.distplot(space_groups)
    plt.savefig('histogram_dist.png')

def _know_space_groups(f):
    dist = np.zeros(230, dtype=int)
    keys = list(f.keys())
    samples = [f[key] for key in keys]
    space_groups = [int(sample.attrs['space_group']) for sample in samples]
    for space_group in space_groups:
        dist[space_group - 1] += 1
    return dist

def iterate_through_data(directory, save_fig=False, fig_name=None):
    '''
    Parameters
    ---------
    directory: String
    save_fig: boolean (optional)
    Whether or not to save a figure visualizing distribution in data
    fig_name: String (optional)
    What to name plot if saved

    Returns
    ---------
    dist_all: Dictionary
    Keys are space groups
    Values are amount of

    '''
    #if save_fig and fig_name=None:
    #    print('No plot name specified, will name plot space_grp_dist')
    #    fig_name = 'space_grp_dist'
    files = sorted([os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.h5')])
    vals = np.zeros(230, dtype=int)
    for file in files:
        try:
            #open a file and run through know_space_groups
            f = h5py.File(file, 'r')
            dist = _know_space_groups(f)
            print('Found distribution of{}'.format(file))
        except OSError:
            print('Could not read {}. Skipping.'.format(file)) 
        vals = np.add(vals, dist)
        #vals = _add(vals, dist)
    keys = np.arange(1, 231, dtype=int)
    dict_dist = {}
    for key,val in zip(keys,vals):
        dict_dist['Space Group {}'.format(key)] = val.item()
    print('Dictionary created.')
    if save_fig:
        if fig_name is None:
            print('No plot name specified, will name plot space_grp_dist')
            fig_name = 'space_grp_dist'
        space_grp = pd.Series(vals, name="Space Group Distribution")
        sns.distplot(space_grp)
        plt.savefig(fig_name + '.png') 
    return dict_dist

def print_space_group_distribution(dict_dist):
'''
Prints out all keys and values in the dictionary of  Space Group distribution
    Parameters
    -------------
    dict_dist: Dictionary object
    Keys are space groups
    Values are amount of  
'''
    for key, val in dict_dist.items():
        print(key, val)
        
def save_space_grp_distribution(dict_dist, file_name='distribution'):
    with open('{}.json'.format(file_name), 'w') as fp:
        json.dump(dict_dist, fp)
'''
# helper function for adding newly found space groups to array of already found
@cuda.jit
def _add(x, y, dist_all):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start, x.shape[0], stride):
        dist_all[i] = x[i] + y[i]

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

    #display_space_group_dist(f)
    show_tree(f)
    dict_dist = iterate_through_data(h5_path, save_fig=True)
    print_space_group_distribution(dict_dist)
    f.close()
'''

