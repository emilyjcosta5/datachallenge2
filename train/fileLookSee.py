# Python script to inspect the structure
# and data stored inside our dataset
# written by Alvin Tan
# 7/13/2019

# load necessary libraries
import h5py
import numpy as np
import pyUSID as usid
from matplotlib import pyplot as plt

# load in the file and check out the
# general structure of the data
h5Path = "batch_train_223.h5"
h5File = h5py.File(h5Path)
print(h5File)
#usid.hdf_utils.print_tree(h5File)
# the file consists of a bunch of groups
# called sample_*_*, each of which hold
# a dataset called cbed_stack

# we can confirm the datatypes
h5Group = h5File["sample_0_0"]
print(h5Group)
h5Main = h5Group["cbed_stack"]
print(h5Main)

# print out the attributes
for attr in h5Group.attrs:
    print("attribute {} is {}".format(attr, usid.hdf_utils.get_attr(h5Group, attr)))

# inspect the cbed_stack
# and print out some graphs
print(h5Main[()].shape)
figs = [None, None, None]
for i in range(3):
    figs[i] = plt.figure()
    plt.suptitle("space group {}, z_dirs {}".format(
              usid.hdf_utils.get_attr(h5Group, "space_group"),
              usid.hdf_utils.get_attr(h5Group, "z_dirs")[0][i]))
    # print out the original image
    plt.subplot(121)
    plt.title("diffusion image")
    plt.imshow(h5Main[()][i]**(1/4))
    plt.colorbar(orientation='vertical')
    # and the fourier transform
    plt.subplot(122)
    plt.title("fourier transform")
    f = np.fft.fft2(h5Main[()][i])
    fshift = np.fft.fftshift(f)
    #magSpec = 20*np.log(np.abs(fshift))
    #plt.imshow(magSpec)
    plt.imshow(np.abs(fshift)**(1/4))
    plt.colorbar(orientation='vertical')
    figs[i].show()
input("Pause to inspect graphs. Press <Enter> to continue...")

# now do the same thing with other data
h5Group = h5File["sample_1_6"]
print(h5Group)
h5Main = h5Group["cbed_stack"]
print(h5Main)

# print out the attributes
for attr in h5Group.attrs:
    print("attribute {} is {}".format(attr, usid.hdf_utils.get_attr(h5Group, attr)))

# inspect the cbed_stack
# and print out some graphs
print(h5Main[()].shape)
figs = [None, None, None]
for i in range(3):
    figs[i] = plt.figure()
    plt.suptitle("space group {}, z_dirs {}".format(
          usid.hdf_utils.get_attr(h5Group, "space_group"),
          usid.hdf_utils.get_attr(h5Group, "z_dirs")[0][i]))
    # print out the original image
    plt.subplot(121)
    plt.title("diffusion image")
    plt.imshow(h5Main[()][i]**(1/4))
    plt.colorbar(orientation='vertical')
    # and the fourier transform
    plt.subplot(122)
    plt.title("fourier transform")
    f = np.fft.fft2(h5Main[()][i])
    fshift = np.fft.fftshift(f)
    #magSpec = 20*np.log(np.abs(fshift))
    #plt.imshow(magSpec)
    plt.imshow(np.abs(fshift)**(1/4))
    plt.colorbar(orientation='vertical')
    figs[i].show()
input("Pause to inspect graphs. Press <Enter> to exit...")















