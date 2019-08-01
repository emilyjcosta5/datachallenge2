'''
Author: Emily Costa
Created on: 08.01.2019
'''

from pandas import DataFrame
import json
import numpy as np

def _convert_JSON_to_arr(j):
    with open(j, 'r') as f:
        data = json.load(f)
        nums = list(range(1,231))
        data = np.array([data['Space Group {}'.format(num)] for num in nums])
    return data

def _add_arr(arr):
    return np.sum(arr)

def find_total_space_grps(keys,js):
    dict = {}
    overall = 0
    for key,j in zip(keys,js):
        arr = _convert_JSON_to_arr(j)
        i = _add_arr(arr)
        overall = overall + i
        dict[key] = str(i)
    dict['Overall'] = str(overall)
    with open('total_space_grps.json', 'w') as f:
        json.dump(dict, f)

if __name__ == '__main__':
    js = ['dataframes/distribution.json', 'dataframes/distributionDev.json', 'dataframes/distributionTest.json']
    keys = ['Train', 'Dev', 'Test']
    find_total_space_grps(keys,js)

