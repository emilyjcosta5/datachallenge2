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


def show_tree(f):
    """
    Displays a tree of the h5.File object
    Groups and datasets
    """
    print("h5 file contains:")
    usid.io.hdf_utils.print_tree(f)


def read_hdf5(f):
    """
    Returns two `dict`s of `dict`s: `images` and `labels`

    image in `images`: 
        key = sample index and 'image_x' string, 
        val = the image in numpy array format

    label in `labels`: 
        key = sample index and its space group, 
        val = the space group in numpy array format
    """
    sets_to_read = ['image_1', 'image_2', 'image_3']
    attrs_to_read = ['space_group']
    keys = list(f.keys())
    images = {} # A dict (key : val) = (sample_index : the image dict)
    labels = {} # A dict (key : val) = (sample_index : the label dict)
    
    # For each sample, extract 3 images and space group as dicts of np.array
    i = 0
    for key in keys:
        if i % 1000 == 0:
            print("Reading HDF5 files: {} - {}".format(i, i+1000))
        sample = f[key]
        images[key] = {s: cbed.tostring() for s, cbed in zip(sets_to_read, sample['cbed_stack'])}
        labels[key] = {a: sample.attrs[a] for a in attrs_to_read}
        i += 1
    f.close()

    return (images, labels)


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_array_feature(array):
    return tf.train.Feature(float_list=tf.train.FloatList(value=array))


def _int_array_feature(array):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=array))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(directory, dataset_name):

    files = sorted([os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.h5')])

    filename = os.path.join(directory, dataset_name + '.tfrecords')
    print('Writing', filename)

    with tf.io.TFRecordWriter(filename) as writer:

        for file in files:
            try:
                f = h5py.File(file, 'r')
                images, labels = read_hdf5(f)
                f.close()
            except OSError:
                print("\tCould not read {}. Skipping.".format(file))
                continue

            print("\tSample size: {}".format(len(images)))
            
            for key in images.keys():
                print("\tProcessing {}...".format(key))
                image = images[key]
                label = labels[key]
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'image_1': _bytes_feature(image['image_1']),
                            'image_2': _bytes_feature(image['image_2']),
                            'image_3': _bytes_feature(image['image_3']),
                            'label': _bytes_feature(label['space_group'])
                        }
                    )
                )
                writer.write(example.SerializeToString())


def reconstruction_test():
    h5_path = os.getcwd() + "/train"
    filename = os.path.join(h5_path, "batch_train_223.h5")
    import matplotlib.pyplot as plt
    
    f = h5py.File(filename, 'r')
    sample = f['sample_0_1']
    # This is the image dataset (array/file)
    cbed_stack = sample['cbed_stack']
    # Display the images
    fig, axes = plt.subplots(2, 3, figsize=(16, 12))
    for ax, cbed in zip(axes[0], cbed_stack):
        print(type(cbed))
        ax.imshow(cbed**0.25)
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
            image = tf.decode_raw(parsed[image_name], tf.uint8)
            height = tf.cast(512, tf.int32)
            width = tf.cast(512, tf.int32)
            image = tf.reshape(image, [height, width, 4])
            return image
        
        image_1 = reshape_image('image_1')
        image_2 = reshape_image('image_2')
        image_3 = reshape_image('image_3')
        label = tf.cast(parsed['label'], tf.string)
        images = {'image_1': image_1, 'image_2': image_2, 'image_3': image_3}
        return images, label

    parsed_dataset = tfdataset.map(_parse_function)
    iterator = parsed_dataset.make_one_shot_iterator()
    images = iterator.get_next()[0]
    for i, ax in enumerate(axes[1]):
        print(i)
        image = images['image_{}'.format(i+1)].numpy()
        ax.imshow(image)
    plt.show()
    

if __name__ == '__main__':
    '''
    h5_path = os.getcwd() + "/train"
    tf.compat.v1.enable_eager_execution()
    
    # convert_to(h5_path, 'train_223')
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
            image = tf.decode_raw(parsed[image_name], tf.uint8)
            height = tf.cast(512, tf.int32)
            width = tf.cast(512, tf.int32)
            image = tf.reshape(image, [height, width, 4])
            return image
        
        image_1 = reshape_image('image_1')
        image_2 = reshape_image('image_2')
        image_3 = reshape_image('image_3')


        # image_1 = tf.image.resize_image_with_crop_or_pad(image_1, height, width)
        # image_2 = tf.image.resize_image_with_crop_or_pad(image_2, height, width)
        # image_3 = tf.image.resize_image_with_crop_or_pad(image_3, height, width)
        label = tf.cast(parsed['label'], tf.string)
        images = {'image_1': image_1, 'image_2': image_2, 'image_3': image_3}

        return images, label


    parsed_dataset = tfdataset.map(_parse_function)
    print(parsed_dataset)
    for parsed_record in parsed_dataset.take(10):
        print(repr(parsed_record))
    
    '''
    tf.compat.v1.enable_eager_execution()
    reconstruction_test()