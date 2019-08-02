# For the Oak Ridge National Lab Data Challenge
# Challenge 2
# written by Emily Costa and Alvin Tan
# 08/02/2019

import os
import numpy as np
import h5py
import random


def _setup_h5_datasets(h5_save_path, fileNames):
    h5Files = []

    # Creates our new .h5 files to be filled out
    for i in range(fileNames):
        filePath = "{}massaged{}.h5".format(h5_save_path, fileNames[i])
        h5Files[i] = h5py.File(filePath)
        print("Created new .h5 file at {}".format(filePath))

    return h5Files


def _distribute_dataset(anH5File, h5Files):
    keys = list(anH5File.keys())
    #samples = [f[key] for key in keys]

    # Pseudorandomly distribute samples among the new files for a
    # roughly even distribution
    for key in keys:
        randNum = random.randrange(len(h5Files))
        anH5File.copy(key, h5Files[randNum])

    return


if __name__ == '__main__':
    # Directory with the current .h5 files
    h5_path = "/gpfs/alpine/world-shared/stf011/junqi/smc/train/"
    # Directory we want to save new .h5 files to. Must end in /
    h5_save_path = ""

    # These are the .h5 files we want to create.
    # Final files will be at h5_save_path + "massaged" + fileNames + ".h5"
    fileNames = ["Train", "Dev", "Test"]

    # Check if we have already made these files before.
    # If so, no need to remake them.
    firstPath = "{}massaged{}.h5".format(h5_save_path, fileNames[0])
    if os.path.exists(firstPath) and os.path.isfile(firstPath):
        print("Files already exist. Ending process.")
        return

    # Create new .h5 files to fill out
    h5Files = _setup_h5_datasets(h5_save_path, fileNames)

    # Iterate through current .h5 files and move distribute the entries
    # into the newly created .h5 files
    curFilePaths = sorted([os.path.join(h5_path, aFileName) for aFileName in os.listdir(directory) if aFileName.endswith('.h5')])

    for aFilePath in curFilePaths:
        try:
            anH5File = h5py.File(aFilePath, 'r')
            _distribute_dataset(anH5File, h5Files)
            print("Distributed {} across new datasets".format(aFilePath))
            anH5File.close()
        except OSError:
            print("Could not read {}. Skipping".format(aFilePath))

    # Closes the newly created .h5 files
    for h5File in h5Files:
        h5File.close()

    print("Finished distribution of all datasets. Newly created datasets can be found in {}".format(h5_save_path))

