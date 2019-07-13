import h5py
import pyUSID as usid


sample_path = 'batch_train_223.h5'
h5_f = h5py.File(sample_path, mode='r+')
usid.io.hdf_utils.print_tree(h5_f)


