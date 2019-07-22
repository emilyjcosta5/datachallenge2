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
from hdf5_to_tfrecord import convert_to
#import pyUSID as usid


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


def show_tree(f):
    """
    Displays a tree of the h5.File object
    Groups and datasets
    """
    print("h5 file contains:")
    usid.io.hdf_utils.print_tree(f)


def reconstruction_test(sample_index=0):
    """Tests the quality of TFRecord conversion
    Displays original 3 images from HDF5 format in the top row
    AND 3 images from TFRecord format in the bottom row
    Args:
        sample_index = takes the `i`th sample
    """
    # Set proper names from the index
    if sample_index < 10:
        sample_name = 'sample_0_{}'.format(sample_index)
    else:
        ind_str = str(sample_index)
        sample_name = 'sample_{}_{}'.format(ind_str[0], ind_str[1])

    h5_path = os.getcwd() + "/train"
    filename = os.path.join(h5_path, "batch_train_223.h5")
    import matplotlib.pyplot as plt
    
    f = h5py.File(filename, 'r')
    sample = f[sample_name]
    cbed_stack = sample['cbed_stack']
    fig, axes = plt.subplots(2, 3, figsize=(16, 12))
    for ax, cbed in zip(axes[0], cbed_stack):
        ax.imshow(cbed**0.25)
    original = [cbed for cbed in cbed_stack]
    f.close()

    filename = os.path.join(h5_path, "train_223.tfrecords")
    tfdataset = tf.data.TFRecordDataset([filename])
    def _parse_function(example_proto):
        # Create a description of the features.
        feature_description = {
            'image_1': tf.io.FixedLenFeature([], dtype=tf.string),
            'image_2': tf.io.FixedLenFeature([], dtype=tf.string),
            'image_3': tf.io.FixedLenFeature([], dtype=tf.string),
            'label': tf.io.FixedLenFeature([], dtype=tf.string, default_value=''),
        }
        parsed = tf.io.parse_single_example(example_proto, feature_description)

        def reshape_image(image_name):
            image = tf.decode_raw(parsed[image_name], tf.int64)
            height = tf.cast(512, tf.int32)
            width = tf.cast(512, tf.int32)
            image = tf.reshape(image, [height, width])
            return image
        
        image_1 = reshape_image('image_1')
        image_2 = reshape_image('image_2')
        image_3 = reshape_image('image_3')
        label = tf.cast(parsed['label'], tf.string)
        images = {'image_1': image_1, 'image_2': image_2, 'image_3': image_3}
        print(images)
        return images, label

    parsed_dataset = tfdataset.map(_parse_function)
    iterator = tf.compat.v1.data.make_one_shot_iterator(parsed_dataset)
    for i in range(sample_index+1):
        # Generate a data record `sample_index` times
        images, label = iterator.get_next()
    for i, ax in enumerate(axes[1]):
        image = images['image_{}'.format(i+1)].numpy()
        ax.imshow(image ** 0.25)

    reconstructed = [image.numpy() for image in images.values()]
    print(original)
    print(reconstructed)
    result = np.allclose(original, reconstructed)
    print("Result:", result)

    label = label.numpy()
    plt.title("Label: " + repr(label))
    plt.show()

if __name__ == '__main__':
    h5_path = "/gpfs/alpine/world-shared/stf011/junqi/smc/train" 
    save_to = "/ccs/home/shutoaraki/challenge_all_data/train" 
    convert_to(h5_path, save_to, 'train_tfrecord')
    tf.compat.v1.enable_eager_execution()
    # Try different indices!
    reconstruction_test(8)
