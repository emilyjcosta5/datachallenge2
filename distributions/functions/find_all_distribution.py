'''
For finding and visualizing distributions from HDF5 files containing Crystallographic imaging data.
Created on: 7.27.2019
Author: Emily Costa
'''
import seaborn as sns
import matplotlib.pyplot as plt
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

def visualize_all_bar(df,headers,colors,file_name='all_dist_hist',ylim=None):
    x = np.arange(1,231)
    #fig, axis = plt.subplots()
    plt.figure(figsize=(20,5))
    plt.style.use('seaborn-darkgrid')
    if not ylim is None:
        plt.ylim(ylim)
    for header,color,arr in zip(headers,colors,arrs):
        print(arr)
        #print(df[header])
        plt.bar(x, arr, color=color,label=header,log=True)
   plt.legend()
    plt.title=("Distribution of Space Groups in Datasets")
    plt.xlabel('Space Group')
    plt.ylabel('Count')
    plt.savefig('{}.png'.format(file_name))

def visualize_all(df,headers,colors,file_name='all_distribution',ylim=None):
    #bins=np.linspace(0,230,1200)
    x = np.arange(1,231)
    #fig, axis = plt.subplots()
    plt.figure(figsize=(18,5))
    plt.style.use('seaborn-darkgrid')
    if not ylim is None:
        plt.ylim(ylim)
    for header,color in zip(headers,colors):
        plt.plot(x,df[header],color=color,alpha=0.5,label=header)
        #plt.hist(df[header],bins,histtype='stepfilled', color=color,alpha=0.3,label=header)
        #sns.distplot(df[header], ax=axis)
    plt.legend()
    plt.title=("Distribution of Space roups in Datasets")
    plt.xlabel('Space Group')
    plt.ylabel('Count')
    
    '''
    n
    sns.distplot(df[headers[0]], color=colors[0])
    sns.distplot(df[headers[1]], color=colors[1])
    sns.distplot(df[headers[2]], color=colors[2])
    sns.distplot(df[headers[3]], color=colors[3])
    
    g = sns.FacetGrid(df, hue=df.index.values)
    g.map(sns.distplot, [df[header] for headers in header]).add_legend()
    g.set(xlabel='Space Groups',ylabel='Count',title='Distribution of Space Groups in All Datasets')
    '''
    plt.savefig('{}.png'.format(file_name))

if __name__ == '__main__':
    files = ['train_redist.json', 'dev_redist.json', 'test_redist.json']
    headers = ['Train redistributed', 'Dev redistributed', 'Test redistributed']
    arrs = [convert_JSON_to_arr(file) for file in files]
    df = create_df(headers,arrs)
    df.describe().to_csv('redist_stat.csv',header=True)
    #save_pd_to_csv(df)
    #for visualizing
    #colors = ['cornflowerblue','m','b']
    #files = ['train_redist.json', 'test_redist.json', 'all_redist.json']
    #headers = ['Train redistributed', 'Dev and test redistributed', 'Train, dev, and test redistributed']
    #visualize_all_bar(arrs,headers,colors,file_name='redistributions_bar_log')
   
