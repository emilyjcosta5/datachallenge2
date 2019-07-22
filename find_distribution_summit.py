'''
Script for finding distribution of space groups in a directory with h5 files then saves to JSON file
Created on: July 22, 2019
Author: Emily Costa
'''

from processing import iterate_through_data, save_space_grp_distribution

h5_path = '/gpfs/alpine/world-shared/stf011/junqi/smc/train/'
dict_dist = iterate_through_data(h5_path)
save_space_grp_distribution(dict_dist)
