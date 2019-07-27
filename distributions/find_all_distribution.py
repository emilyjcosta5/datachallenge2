'''
For finding and visualizing distributions from HDF5 files containing Crystallographic imaging data.
Created on: 7.27.2019
Author: Emily Costa
'''
import seaborn as sns
import matplotlib.pyplot as pl
import pandas as pd
from pandas import DataFrame
import json
import numpy as np

def convert_JSON_to_arr(j):
    with open(j, 'r') as f:
        data = json.load(f)
        nums = list(range(1,231))
        data = np.array([data['Space Group {}'.format(num)] for num in nums])
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

def create_df(headers,arrs):
    '''
    arrs: List of Dictionaries
    keys are headers of data, strings
    vals is data, numpy array
    '''
    #overall = []
    #headers = []
    #for key,val in zip(header,arrs):
    #    overall.append(arr.value())
    #    headers.append(arr.keys())
    data_dict = dict(zip(headers,arrs))
    grps = list(range(1,231))
    df = DataFrame(data_dict,index=['Space Group {}'.format(grp) for grp in grps])
    return df

def save_pd_to_csv(df):
    df.to_csv('distribution.csv', header=True)

def describe_data(df,file_name='description'):
    description = df.describe()
    description.to_csv('{}.csv'.format(file_name), header=True)

def visualize_all(df,headers,colors,file_name='all_distributions'):
    sns.distplot([df[header] for header in headers], color=colors)
    plt.savefig('{}.png'.format(file_name))

if __name__ == '__main__':
     files = ['overall_distribution.json', 'distribution.json', 'distributionDev.json', 'distributionTest.json']
     arrs = [convert_JSON_to_arr(file) for file in files]
     headers = ['Overall', 'Train', 'Dev', 'Test']
     colors = ['blue','m','mediumvioletred','chartreuse']
     df = create_df(headers,arrs)
     visualize_all(df, headers,colors)
    
