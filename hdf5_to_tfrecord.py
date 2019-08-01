#!/usr/bin/env python

# @author: Shuto Araki
# @date: 07/16/2019
# ==============================================================================

"""Converts HDF5 data files to TFRecords file format with Example protos."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf
import numpy as np
import h5py

FLAGS = None

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
        print("Processing", key)
        if i % 10 == 0: 
            print("Reading HDF5 files: {} - {}".format(i, i+10))
        sample = f[key]
        # numpy.tostring() is a lossy compression for float data (maybe)
        squish = lambda cbed: np.array([np.interp(cb, (np.amin(cbed), np.amax(cbed)), (0, 255)) for cb in cbed]).astype(int)
        images[key] = {s: squish(cbed).tostring() for s, cbed in zip(sets_to_read, sample['cbed_stack'])}
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


def convert_to(from_dir, to_dir, dataset_name, start):

    files = sorted([os.path.join(from_dir, file) for file in os.listdir(from_dir) if file.endswith('.h5')])
    files = files[start:start+2] # For testing on Summit
    filename = os.path.join(to_dir, dataset_name + '.tfrecords')
    print(files)
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
                label = int(label)
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'image_1': _bytes_feature(image['image_1']),
                            'image_2': _bytes_feature(image['image_2']),
                            'image_3': _bytes_feature(image['image_3']),
                            'label': _int64_feature(label['space_group'])
                        }
                    )
                )
                writer.write(example.SerializeToString())


def main(unused_argv):
    i = 0
    while i < 2: # Just one tfrecord file (2 HDF-5 files) for testing
        try:
            convert_to(from_dir=FLAGS.train_dir, to_dir='/mnt/bb/shutoaraki/', dataset_name='train-{:05d}-of-01024'.format(i), start=i)
            i += 2 
        except:
            print("Done processing {} files!".format(i))
            break
    # convert_to(FLAGS.train_dir, './data/val', 'val_tfrecord')


if __name__ == '__main__':

    def is_valid_folder(x):
        """
        'Type' for argparse - checks that file exists but does not open.
        """
        x = os.path.expanduser(x)
        if not os.path.isdir(x):
            raise argparse.ArgumentTypeError("{0} is not a directory".format(x))
        return x

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', metavar='directory', type=is_valid_folder)
    # parser.add_argument('--val_dir', metavar='directory', type=is_valid_folder)

    FLAGS, unparsed = parser.parse_known_args()
    tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)
