import tarfile
from six.moves import urllib
import sys
import numpy as np
import pickle
import os
import cv2
import h5py as h5

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_DEPTH = 2
NUM_CLASS = 3

EPOCH_SIZE = 46160

def load(paths=None):
    """Load and reshape data from file."""

    if paths is None:
        # change paths here
        paths = {'A': 'C:\code\cremi\deep_res_net\data\\test_like_training_all_A_good_masks_augmented.h5',
                 'B': 'C:\code\cremi\deep_res_net\data\\test_like_training_all_B_good_masks_augmented.h5',
                 'C': 'C:\code\cremi\deep_res_net\data\\test_like_training_all_C_good_masks_augmented.h5'}

        #paths = {'A': '/net/hciserver03/storage/plitvak/resnet/resnet-in-tensorflow/data/test_like_training_all_A_good_masks_augmented.h5',
        #       'B': '/net/hciserver03/storage/plitvak/resnet/resnet-in-tensorflow/data/test_like_training_all_B_good_masks_augmented.h5',
        #        'C': '/net/hciserver03/storage/plitvak/resnet/resnet-in-tensorflow/data/test_like_training_all_C_good_masks_augmented.h5'}

       # paths = {'A': 'C:\code\cremi\deep_res_net\data\\test_like_training_all_A_good_masks_augmented.h5'}

    print("path, ", paths.values())
    if not all([os.path.exists(path) for path in paths.values()]):
        raise IOError("Data files not found. Please download them, \n"
                      "and make sure they're named 'training_all_[A or B or C]_2.h5'.")

    # Init a dict to contain arrays for all datasets
    datadict = {'trX': [], 'vaX': [],
                'trXm': [], 'vaXm': [],
                'trY': [], 'vaY': []}

    # Loop over datasets and load from H5
    for dset in paths.keys():
        # Train
        h5file = h5.File(paths[dset], "r")

        A = np.moveaxis(h5file['images_50_end'], 2, 0)
        datadict['trX'].append(A)
        M = np.moveaxis(h5file['masks_50_end'], 2, 0)
        datadict['trXm'].append(M)
        Y = np.moveaxis(np.asarray(h5file['labels_50_end']).reshape(-1), 0, 0)
        datadict['trY'].append(Y)

        # Validate
        A = np.moveaxis(h5file['images_0_50'], 2, 0)
        datadict['vaX'].append(A)
        M = np.moveaxis(h5file['masks_0_50'], 2, 0)
        datadict['vaXm'].append(M)
        Y = np.moveaxis(np.asarray(h5file['labels_0_50']).reshape(-1), 0, 0)
        datadict['vaY'].append(Y)

    # Concatenate to small tensors bigass tensors
    datadict = {key: np.concatenate(tuple(arr), axis=0) for key, arr in datadict.items()}

    return datadict


def whitening_image(X, Xm, shuffle=False):
    '''
    Performs per_image_whitening
    :param image_np: a 4D numpy array representing a batch of images
    :return: the image numpy array after whitened
    '''
    ni = X.shape[0]
    idx = np.arange(ni)
    if shuffle:
        np.random.shuffle(idx)

    # im2double on X
    X = X.astype('float32')
    X *= 1. / 255.
    # Center Xm to 0.5
    #Xm = Xm.astype('float32')
    # Xm -= 0.5

    #Y = Y.astype('float32')

    # Get mean and std of the dataset, and center the X values.
    meanX = X.mean()
    stdX = X.std()
    print(meanX)
    print(stdX)
    X = (X - meanX) / stdX

    X = np.expand_dims(X, -1)
    Xm = np.expand_dims(Xm, -1)
    X = np.multiply(X, Xm)
    X = np.concatenate((X, Xm - 0.5), axis=3)

    return X


def prepare_train_data(dataset, padding_size):
    '''
    Read all the train data into numpy array and add padding_size of 0 paddings on each side of the
    image
    :param padding_size: int. how many layers of zero pads to add on each side?
    :return: all the train data and corresponding labels
    '''

    data, masks, label = dataset['trX'], dataset['trXm'], dataset['trY']
    data = whitening_image(data, masks)

    return data, label

def read_validation_data(dataset):
    '''
    Read in validation data. Whitening at the same time
    :return: Validation image data as 4D numpy array. Validation labels as 1D numpy array
    '''
    validation_array, validation_masks, validation_labels = dataset['vaX'], dataset['vaXm'], dataset['vaY']

    validation_array = whitening_image(validation_array, validation_masks)
    return validation_array, validation_labels
