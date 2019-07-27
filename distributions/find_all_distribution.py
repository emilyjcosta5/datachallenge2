'''
For finding and visualizing distributions from HDF5 files containing Crystallographic imaging data.
Created on: 7.27.2019
Author: Emily Costa
'''

import json
import numpy as np

def convert_JSON_to_dict(j):
    with open(j, 'r') as f:
        data = json.load(f)
        data = np.array(list(data.values()))
    return data
        

def make_JSON_from_Dicts(dict_arrs, file_name='overall_distribution'):
    '''
    Writes a JSON file containing overall distribution of space groups.

    Parameters
    --------
    dict_files: tuple of numpy arrays
    distribution of multiple space groups
    file_name: (optional) String
    name for file to be written
    '''
    total = np.zeros(230, dtype=int) 
    for arr in dict_arrs:
        print(arr)
        total = np.add(total, arr)
    keys = np.arange(1,231,dtype=int)
    dict_all = {}
    for key,val in zip(keys,total):
        dict_all['Space Group {}'.format(key)]=val.item()
    with open('{}.json'.format(file_name), 'w') as fp:
        json.dump(dict_all, fp)

if __name__ == '__main__':
    all = [convert_JSON_to_dict('distribution.json'),convert_JSON_to_dict('distributionDev.json'),convert_JSON_to_dict('distributionTest.json')]
    make_JSON_from_Dicts(all)
