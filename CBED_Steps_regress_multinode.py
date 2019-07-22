
# coding: utf-8

# # Deep Learning for Inverse Imaging: Step Determination
# 
# #### M. Oxley, J. Yin, S. Jesse, N. Borodinov, A. Lupini, R. Vasudevan
# 
# #### Notebook by R. Vasudevan and J. Yin
# 
# Latest Date: June 23rd 2018
# Here we are trying to determine the precise position of the step, down to the nearest 10% 
# Thus it remains a classification problem

from __future__ import division
import os
import re
from random import shuffle
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from skimage.transform import rotate

#Load all keras functions
from keras.models import Sequential
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam, SGD
from keras.models import load_model
from keras.regularizers import l2
from keras.models import Sequential, load_model
from keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization, AveragePooling3D
from keras.utils.training_utils import multi_gpu_model
import horovod.keras as hvd 
import tensorflow as tf
from keras import backend as K
import keras 




def return_filenames(folder_path, file_id_percent = 0.10, file_id_test_percent = 0.20, thickness = '100A'):
    #file_id_percent: The % of files to withold, for later validation. 
    
    thicknesses = os.listdir(folder_path) #Different thickness values
       
    #Get the filenames
    fnames_steps = []
    #for thickness in thicknesses:
    folder_path_c1  = os.path.join(folder_path, thickness, r'steps')

    file_names_steps = os.listdir(folder_path_c1)

    for filen in file_names_steps:
        if filen.endswith('.npy'):
            fnames_steps.append(os.path.join(folder_path_c1, filen))

    #Shuffle them u
    shuffle(fnames_steps)
    
    #Maintain class balance by taking the same number of diffusion files as step files.
    file_id_max = int((1-file_id_percent)*len(fnames_steps))
    file_id_test_max = int((1-file_id_test_percent)*len(fnames_steps))
    
    fnames_steps_test = fnames_steps[0:file_id_test_max]
    fnames_steps_train = fnames_steps[file_id_test_max:file_id_max]
    fnames_steps_valid = fnames_steps[file_id_max:]
    
    #Shuffle up the filenames

    shuffle(fnames_steps_test)
    shuffle(fnames_steps_train)
    shuffle(fnames_steps_valid)
    
    print('Number of files for training: ' + str(len(fnames_steps_train)))
    print('Number of files for testing: ' + str(len(fnames_steps_test)))
    print('Number of files for validation: ' + str(len(fnames_steps_valid)))
    return fnames_steps_train, fnames_steps_test, fnames_steps_valid

#Simple function to get the maximum step number for each thickness
def getMaxSteps(folder_path):
    #Given the folder path containing simulation results split by thicknesses,
    #Look through each thickness and extract the max position of the step
    #Return the result as a dictionary
    
    max_step_dict = dict()
    for thickness in thicknesses:
        max_step = 0
        file_list = os.listdir(os.path.join(folder_path, thickness, r'steps'))
        for fname in file_list:
            if '.npy' in fname:
                name = re.search(r'STO_LAO_\D+_.*_pp', fname).group(0).split('_')
                step_num = int(name[3])
                if step_num>max_step: max_step = step_num

        max_step_dict[thickness] = max_step
    return max_step_dict

#Now we need our generator again!
#Generator for images. Gievn a list of filenames, split into batches and return train/test splits.

def get_labels(dfile, max_step_dict):
    max_step = max_step_dict[dfile.split('/')[-3]]
    name = re.search(r'STO_LAO_\D+_.*_pp', dfile).group(0).split('_')
    step_num = int(name[3])
    percentage_pos = step_num / max_step*10
    label_val = int(np.round(percentage_pos,0))
    return label_val

def myGenerator(filenames_list,max_step_dict, batch_num=0, batch_size=3): 
    #the arguments passed to the generator object is ONLY used during the first call! 
    '''Inputs:
        - Filenames_list: list of filenames ot pass to the generator. This is produced 
        via the returnFilenames() function
        - max_step_dict: This is necessary to return the correct labels. Again, this
        is produced via the getMaxSteps() function
        - batch_num: int, where to start the generator (generally 0)
        - batch_size: int, how many files per batch to output from the generator
        
    Returns:
        - X_train, y_train, X_test, y_test: train/test split, 
        with one-hot vectors for the labels are returned
    '''
    
    while True:
        if batch_num*batch_size>=len(filenames_list):
            break
        print('Batch: ' + str(batch_num))
        
        X_data = []
        y_data = []
        fnames_batch = filenames_list[batch_num*batch_size:batch_num*batch_size+batch_size]
        
        #Return the data, append to a matrix
        for dfile in fnames_batch:
            training_data = np.load(dfile).mean(axis=0)
            
            #Let's go through and add noise and rotations
            training_data_aug = np.zeros(shape=training_data.shape)
            
            rotation_val = np.random.uniform(low=-5, high = 5) #rotation angle
            noise_k = np.random.uniform(low = 0.01, high = 0.05)#poisson noise factor
            for i in range(training_data.shape[0]):
                img = training_data[i,:,:]
                #Rotate
                img_rotated = rotate(img, angle =rotation_val, resize=False, mode = 'edge' )
                #Add Poisson noise
                poisson_noise = np.random.poisson(img_rotated)
                img_noised = img_rotated + noise_k*poisson_noise 
                training_data_aug[i,:,:] = img_noised 
            
            X_data.append(training_data_aug)
            #Return the label matrix. Generate it on the fly.
            y_data.append(get_labels(dfile, max_step_dict))
            
        #Reshaping to collapse first axis    
        X_data = np.array(X_data)
        X_data = X_data#.reshape(X_data.shape[0],X_data.shape[1], X_data.shape[2], X_data.shape[3])
        y_data = np.array(y_data)
        
        #Convert y_data to one-hot encoding
        onehot_encoder = OneHotEncoder(sparse=False, n_values=11)
        y_data_oh = onehot_encoder.fit_transform(y_data.reshape(-1,1))
        
        #Do a test/train split
        #X_train, X_test, y_train, y_test = train_test_split(X_data, y_data_oh, test_size=0.20, 
        #                                                    shuffle = True) 
        #yield X_train, y_train, X_test, y_test
	yield X_data, y_data_oh
        batch_num=batch_num+1 #remember after yield the generator will return to this point when next() is called.


def plot_sequence_images(label, image, title = 1):
    #Plot label as title with 3D image sequence
    num_cols = 8
    
    fig, axes = plt.subplots(nrows=1, ncols = num_cols, figsize = (20,5))
    for ind, ax in enumerate(axes.flat):
        img_ind = int((ind/num_cols)*image.shape[0])
        ax.imshow(image[img_ind,:,:])
        if title!=0:
            ax.set_title(str(label))
        ax.axis('off')
  
    return fig


hvd.init()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())
K.set_session(tf.Session(config=config))



thickness = '200A'
folder_path = r'./simulations/results'
thicknesses = os.listdir(folder_path)

#Get the max step number for each thickness, store in a dict
max_step_dict = getMaxSteps(folder_path)

#Now get a train/test split of filenames
steps_file_train, steps_file_test, steps_file_valid = return_filenames(folder_path, thickness=thickness)

data_gen = myGenerator(steps_file_train, max_step_dict, batch_num=0, batch_size=5) #initialize the generator object
X_train, y_train = data_gen.next() 

#Setup the simple Keras DCNN classifier
num_classes = 11 #binary classifier

input_shape = X_train.shape[1:]
    
model = Sequential()

model.add(Conv3D(32, kernel_size=(1,5,5), 
                 activation='relu',
                 input_shape=(input_shape[0],input_shape[1],input_shape[2],1),
         kernel_regularizer=l2(0.01))) #1

model.add(Dropout(0.15))
model.add(Conv3D(64, (2, 5, 5), activation='relu', kernel_regularizer=l2(0.01))) # 2

model.add(Dropout(0.15))
model.add(Conv3D(64, (2, 5, 5), activation='relu', kernel_regularizer=l2(0.01))) # 3

model.add(AveragePooling3D(pool_size = (2,2,2)))

model.add(Flatten())

model.add(Dropout(0.15))

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.15))

model.add(Dense(256, activation='relu'))

model.add(Dense(num_classes, activation='softmax'))

opt = keras.optimizers.SGD(0.01*hvd.size())

opt = hvd.DistributedOptimizer(opt)
model.compile(loss=categorical_crossentropy,
              optimizer=opt,
              metrics=['accuracy'])

#gpu_model = multi_gpu_model(model, gpus=2)

#gpu_model.compile(loss=categorical_crossentropy,
#              optimizer=SGD(),
#              metrics=['accuracy'])

epochs = 50 #Number of epochs
batch_size = 16 #Batch size
callbacks = [ hvd.callbacks.BroadcastGlobalVariablesCallback(0),]

if hvd.rank() == 0:
    callbacks.append(keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'))


data_gen = myGenerator(steps_file_train, max_step_dict, batch_num=0, batch_size=batch_size)
test_gen = myGenerator(steps_file_test, max_step_dict, batch_num=0, batch_size=batch_size)
model.fit_generator(data_gen, 
		    steps_per_epoch=len(data_gen) // hvd.size(),
		    callbacks=callbacks,
		    epochs=epochs,
		    verbose=1,
		    worker=4,
		    initial_epoch=resume_from_epoch,
		    validation_data=test_gen,
		    validation_steps=len(test_gen) // hvd.size())
score = hvd.allreduce(model.evaluate_generator(test_gen,len(test_gen),workers=4))
print('Test loss:', score[0])
print('Test accuracy', score[1])
# In[11]:

#if hvd.rank() == 0:
#    training_stats = np.array(training_stats)
#    np.save('training_stats.npy', training_stats)


# In[12]:

