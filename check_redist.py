import h5py
from distributions.functions.find_total_in_set import *
from processing.processing import *

js = ['processing/massagedDev.json', 'processing/massagedTest.json', 'processing/massagedTrain.json']
keys = ['Dev','Test','Train']

dev_path = '/gpfs/alpine/gen011/scratch/ecost020/datachallenge2/processing/massagedDev.h5'
train_path = '/gpfs/alpine/gen011/scratch/ecost020/datachallenge2/processing/massagedTrain.h5'
test_path = '/gpfs/alpine/gen011/scratch/ecost020/datachallenge2/processing/massagedTest.h5'

dict_dev = iterate_through_data(dev_path)
save_space_grp_distribution(dict_dev, file_name='redistDev')

dict_train = iterate_through_data(train_path)
save_space_grp_distribution(dict_train, file_name='redistTrain')

dict_test = iterate_through_data(test_path)
save_space_grp_distribution(dict_test, file_name='redistTest')
