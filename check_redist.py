import h5py
from distributions.functions.find_total_in_set import *
from processing.processing import *

js = ['processing/massagedDev.json', 'processing/massagedTest.json', 'processing/massagedTrain.json']
keys = ['Dev','Test','Train']

dev_path = '/gpfs/alpine/world-shared/stf011/junqi/smc/dev'
train_path = '/gpfs/alpine/world-shared/stf011/junqi/smc/train'
test_path = '/gpfs/alpine/world-shared/stf011/junqi/smc/test'
