# -*- coding: utf-8 -*-
# torch packages
import torch
import torchvision
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from torch import optim
from torchvision import datasets, transforms, models
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms.functional as TF


from torchmetrics import MetricCollection
from torchmetrics.classification import MultilabelRecall, MultilabelPrecision, MultilabelF1Score

import torch.nn.functional as F
from torch.utils.data import Dataset


# additional packages
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import argparse
import os
import time
import glob
from tqdm import tqdm



class QUAM(data.Dataset):
    ''' Dataset object for the QUAM-AFM dataset'''
    def __init__(self, args, dataset_df, mode = 'train', dimensions=None):
        if dimensions is None:
            dimensions = [224, 224] #Default input size for Resnet
        self.args = args
        self.data = dataset_df
        self.dimensions = dimensions
        self.mode = mode
        if len(self.data) == 0:
            raise RuntimeError('Found 0 images, please check the data set')


    def get_stack(self, path):
        '''Obtains stack of images from folder path
            INPUT: path of the folder containing the AFM images of the molecule
            OUTPUT: list of PIL images'''

        last_str = path.split('/')[-1]
        im_list = list()
        for i in range(10):
            im_path = path + '/' + last_str + f'_df_00{i}.jpg'
            im_list.append(Image.open(im_path).resize(self.dimensions, resample=Image.Resampling.BILINEAR))

        return im_list

    def data_aug(self, im_list):

        'Perform data augmentations'
        im_stack = im_list.copy()

        # Default parameters
        zooming = 1
        degrees = 0
        h_shift = 0
        v_shift = 0
        shear = 0

        # Zoom
        if np.random.uniform() < self.args.zoom_prob:
            zooming = np.random.uniform(low=(1 - self.args.max_zoom), high=1 + self.args.max_zoom)

        # Rotation
        if np.random.uniform() < self.args.rot_prob:
            degrees = np.random.uniform(low=-self.args.max_deg, high=self.args.max_deg)

        # Translation:
        if np.random.uniform() < self.args.shift_prob:
            h_shift, v_shift = np.random.randint(0, high=self.args.max_shift, size=2, dtype=int)

        # Shear
        if np.random.uniform() < self.args.shear_prob:
            shear = np.random.uniform(low=-self.args.max_shear, high=self.args.max_shear)

        for i in range(len(im_stack)):
            im_stack[i] = TF.affine(im_stack[i], degrees, [h_shift, v_shift], zooming, shear,
                                    interpolation=transforms.InterpolationMode.BILINEAR,
                                    fill=im_stack[i].getpixel((5, 5))
                                    )

        return im_stack

    def __getitem__(self, index):

        # We should have different preprocessings for train and test.
        # We could perform data augmentations. For now we only leave a reminder.

        img_path = self.data.iloc[index]['path']
        im_stack = self.get_stack(img_path)  # list of PIL images

        if self.mode == 'train':
            im_stack = self.data_aug(im_stack)

        to_tensor = transforms.ToTensor()

        tensor_stack = [to_tensor(im) for im in im_stack.copy()]
        tensor_stack = torch.cat(tensor_stack)

       #normalize = transforms.Normalize(
        #    mean=[0.1321, 0.1487, 0.1689, 0.1983, 0.2229, 0.2591, 0.3430, 0.4580, 0.5787, 0.6887],
        #    std=[0.0853, 0.0853, 0.0883, 0.0942, 0.0969, 0.1066, 0.1419, 0.1840, 0.2144, 0.2215]) #this is for K-1



        normalize = transforms.Normalize(
            mean=[0.2855, 0.3973, 0.4424, 0.4134, 0.6759, 1.0664, 0.9904, 0.7708, 0.5748, 0.4055],
            std=[1.1341, 1.2528, 1.3125, 1.3561, 1.5844, 1.7763, 1.5447, 1.2683, 1.0588, 0.9308]) # for all K folders

        tensor_stack = normalize(tensor_stack)

        labels = self.data.iloc[index]['morgan_fp']
        labels = torch.from_numpy(labels)

        return tensor_stack, labels.type(torch.FloatTensor)

    def __len__(self):
        return len(self.data)





class QUAM_with_noise(data.Dataset):
    ''' Dataset object for the QUAM-AFM dataset with gaussian noise augmentation'''
    def __init__(self, args, dataset_df, mode = 'train', dimensions=None):
        if dimensions is None:
            dimensions = [224, 224] #Default input size for Resnet
        self.args = args
        self.data = dataset_df
        self.dimensions = dimensions
        self.mode = mode
        if len(self.data) == 0:
            raise RuntimeError('Found 0 images, please check the data set')


    def get_stack(self, path):
        '''Obtains stack of images from folder path

            INPUT: path of the folder containing the AFM images of the molecule

            OUTPUT: list of PIL images'''

        last_str = path.split('/')[-1]
        im_list = list()
        for i in range(10):
            im_path = path + '/' + last_str + f'_df_00{i}.jpg'
            im_list.append(Image.open(im_path).resize(self.dimensions, resample=Image.Resampling.BILINEAR))

        return im_list

    def add_gaussian_noise(self, img_pil, mean=0, std=0, dimensions=[224,224]):
        '''Adds gaussian noise to PIL image'''
        img_arr = np.asarray(img_pil)
        img_arr = np.minimum(np.maximum(0, img_arr+np.random.normal(loc=mean, scale=std, size=dimensions)), 255)

        return Image.fromarray(img_arr).convert('L')

    def data_aug(self, im_list):

        'Perform data augmentations'
        im_stack = im_list.copy()

        # Default parameters
        zooming = 1
        degrees = 0
        h_shift = 0
        v_shift = 0
        shear = 0

        # Zoom
        if np.random.uniform() < self.args.zoom_prob:
            zooming = np.random.uniform(low=(1 - self.args.max_zoom), high=1 + self.args.max_zoom)

        # Rotation
        if np.random.uniform() < self.args.rot_prob:
            degrees = np.random.uniform(low=-self.args.max_deg, high=self.args.max_deg)

        # Translation:
        if np.random.uniform() < self.args.shift_prob:
            h_shift, v_shift = np.random.randint(0, high=self.args.max_shift, size=2, dtype=int)

        # Shear
        if np.random.uniform() < self.args.shear_prob:
            shear = np.random.uniform(low=-self.args.max_shear, high=self.args.max_shear)

        for i in range(len(im_stack)):

            im_stack[i] = TF.affine(self.add_gaussian_noise(im_stack[i],  std=self.args.gauss_noise,
                                                            dimensions=self.dimensions),
                                    degrees, [h_shift, v_shift], zooming, shear,
                                    interpolation=transforms.InterpolationMode.BILINEAR,
                                    fill=im_stack[i].getpixel((5, 5))
                                    )

        return im_stack

    def __getitem__(self, index):

        # We should have different preprocessings for train and test.
        # We could perform data augmentations. For now we only leave a reminder.

        img_path = self.data.iloc[index]['path']
        im_stack = self.get_stack(img_path)  # list of PIL images

        if self.mode == 'train':
            im_stack = self.data_aug(im_stack)

        to_tensor = transforms.ToTensor()

        tensor_stack = [to_tensor(im) for im in im_stack.copy()]
        tensor_stack = torch.cat(tensor_stack)

        #normalize = transforms.Normalize(
        #    mean=[0.1321, 0.1487, 0.1689, 0.1983, 0.2229, 0.2591, 0.3430, 0.4580, 0.5787, 0.6887],
        #    std=[0.0853, 0.0853, 0.0883, 0.0942, 0.0969, 0.1066, 0.1419, 0.1840, 0.2144, 0.2215]) #this is for K-1



        normalize = transforms.Normalize(
            mean=[0.2855, 0.3973, 0.4424, 0.4134, 0.6759, 1.0664, 0.9904, 0.7708, 0.5748, 0.4055],
            std=[1.1341, 1.2528, 1.3125, 1.3561, 1.5844, 1.7763, 1.5447, 1.2683, 1.0588, 0.9308]) # for all K folders


        mean: tensor()
        tensor_stack = normalize(tensor_stack)

        labels = self.data.iloc[index]['morgan_fp']
        labels = torch.from_numpy(labels)

        return tensor_stack, labels.type(torch.FloatTensor)

    def __len__(self):
        return len(self.data)


class QUAM_atom_regressor_10_imgs(QUAM_with_noise):
    ''' Dataset object for the QUAM-AFM dataset atom frequency regression task.
        This dataloader outputs the 10 imgs stack and the atom counts for each molecule
        with format ['C', 'Br', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S', 'H'].
        It inherits from QUAM_with_noise, so it includes gaussian noise augmentation.

        Example: from image stack of molecule C22H13N3O4:
        output =  [22, 0, 0, 0, 0, 3, 4, 0, 0, 13]'''


    def __getitem__(self, index):


        img_path = self.data.iloc[index]['path']
        im_stack = self.get_stack(img_path)  # list of PIL images

        if self.mode == 'train':
            im_stack = self.data_aug(im_stack)

        to_tensor = transforms.ToTensor()

        tensor_stack = [to_tensor(im) for im in im_stack.copy()]
        tensor_stack = torch.cat(tensor_stack)

        #normalize = transforms.Normalize(
        #    mean=[0.1321, 0.1487, 0.1689, 0.1983, 0.2229, 0.2591, 0.3430, 0.4580, 0.5787, 0.6887],
        #    std=[0.0853, 0.0853, 0.0883, 0.0942, 0.0969, 0.1066, 0.1419, 0.1840, 0.2144, 0.2215]) #this is for K-1



        normalize = transforms.Normalize(
            mean=[0.2855, 0.3973, 0.4424, 0.4134, 0.6759, 1.0664, 0.9904, 0.7708, 0.5748, 0.4055],
            std=[1.1341, 1.2528, 1.3125, 1.3561, 1.5844, 1.7763, 1.5447, 1.2683, 1.0588, 0.9308]) # for all K folders


        tensor_stack = normalize(tensor_stack)

        labels = self.data.iloc[index][['C', 'Br', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S', 'H']]

        return tensor_stack, torch.FloatTensor(labels)






class QUAM_3_imgs(data.Dataset):
    ''' Dataset object for the QUAM-AFM dataset with gaussian noise augmentation. It only outputs
    a tomography with 3 images, at heights 1,4,7'''
    def __init__(self, args, dataset_df, mode = 'train', dimensions=None):
        if dimensions is None:
            dimensions = [224, 224] #Default input size for Resnet
        self.args = args
        self.data = dataset_df
        self.dimensions = dimensions
        self.mode = mode
        if len(self.data) == 0:
            raise RuntimeError('Found 0 images, please check the data set')


    def get_stack(self, path):
        '''Obtains stack of images from folder path

            INPUT: path of the folder containing the AFM images of the molecule

            OUTPUT: list of PIL images'''

        last_str = path.split('/')[-1]
        im_list = list()
        for i in self.args.img_heights:
            im_path = path + '/' + last_str + f'_df_00{i}.jpg'
            im_list.append(Image.open(im_path).resize(self.dimensions, resample=Image.Resampling.BILINEAR))

        return im_list

    def add_gaussian_noise(self, img_pil, mean=0, std=0, dimensions=[224,224]):
        '''Adds gaussian noise to PIL image'''
        img_arr = np.asarray(img_pil)
        img_arr = np.minimum(np.maximum(0, img_arr+np.random.normal(loc=mean, scale=std, size=dimensions)), 255)

        return Image.fromarray(img_arr).convert('L')

    def data_aug(self, im_list):

        'Perform data augmentations'
        im_stack = im_list.copy()

        # Default parameters
        zooming = 1
        degrees = 0
        h_shift = 0
        v_shift = 0
        shear = 0

        # Zoom
        if np.random.uniform() < self.args.zoom_prob:
            zooming = np.random.uniform(low=(1 - self.args.max_zoom), high=1 + self.args.max_zoom)

        # Rotation
        if np.random.uniform() < self.args.rot_prob:
            degrees = np.random.uniform(low=-self.args.max_deg, high=self.args.max_deg)

        # Translation:
        if np.random.uniform() < self.args.shift_prob:
            h_shift, v_shift = np.random.randint(0, high=self.args.max_shift, size=2, dtype=int)

        # Shear
        if np.random.uniform() < self.args.shear_prob:
            shear = np.random.uniform(low=-self.args.max_shear, high=self.args.max_shear)

        for i in range(len(im_stack)):

            im_stack[i] = TF.affine(self.add_gaussian_noise(im_stack[i],  std=self.args.gauss_noise,
                                                            dimensions=self.dimensions),
                                    degrees, [h_shift, v_shift], zooming, shear,
                                    interpolation=transforms.InterpolationMode.BILINEAR,
                                    fill=im_stack[i].getpixel((5, 5))
                                    )

        return im_stack

    def __getitem__(self, index):

        # We should have different preprocessings for train and test.
        # We could perform data augmentations. For now we only leave a reminder.

        img_path = self.data.iloc[index]['path']
        im_stack = self.get_stack(img_path)  # list of PIL images

        if self.mode == 'train':
            im_stack = self.data_aug(im_stack)

        to_tensor = transforms.ToTensor()

        tensor_stack = [to_tensor(im) for im in im_stack.copy()]
        tensor_stack = torch.cat(tensor_stack)

        normalize = transforms.Normalize(
            mean=[0.1487, 0.2229, 0.4580],
            std=[0.0853, 0.0969, 0.1840])

        tensor_stack = normalize(tensor_stack)

        labels = self.data.iloc[index]['morgan_fp']
        labels = torch.from_numpy(labels)

        return tensor_stack, labels.type(torch.FloatTensor)

    def __len__(self):
        return len(self.data)






class QUAM_1_img(data.Dataset):
    ''' Dataset object for the QUAM-AFM dataset with gaussian noise augmentation. It outputs 1
    image at height 5, repeated 3 times'''
    def __init__(self, args, dataset_df, mode = 'train', dimensions=None):
        if dimensions is None:
            dimensions = [224, 224] #Default input size for Resnet
        self.args = args
        self.data = dataset_df
        self.dimensions = dimensions
        self.mode = mode
        if len(self.data) == 0:
            raise RuntimeError('Found 0 images, please check the data set')


    def get_stack(self, path):
        '''Obtains stack of images from folder path

            INPUT: path of the folder containing the AFM images of the molecule

            OUTPUT: list of PIL images'''

        last_str = path.split('/')[-1]
        im_list = list()
        for i in [4,4,4]:
            im_path = path + '/' + last_str + f'_df_00{i}.jpg'
            im_list.append(Image.open(im_path).resize(self.dimensions, resample=Image.Resampling.BILINEAR))

        return im_list

    def add_gaussian_noise(self, img_pil, mean=0, std=0, dimensions=[224,224]):
        '''Adds gaussian noise to PIL image'''
        img_arr = np.asarray(img_pil)
        img_arr = np.minimum(np.maximum(0, img_arr+np.random.normal(loc=mean, scale=std, size=dimensions)), 255)

        return Image.fromarray(img_arr).convert('L')

    def data_aug(self, im_list):

        'Perform data augmentations'
        im_stack = im_list.copy()

        # Default parameters
        zooming = 1
        degrees = 0
        h_shift = 0
        v_shift = 0
        shear = 0

        # Zoom
        if np.random.uniform() < self.args.zoom_prob:
            zooming = np.random.uniform(low=(1 - self.args.max_zoom), high=1 + self.args.max_zoom)

        # Rotation
        if np.random.uniform() < self.args.rot_prob:
            degrees = np.random.uniform(low=-self.args.max_deg, high=self.args.max_deg)

        # Translation:
        if np.random.uniform() < self.args.shift_prob:
            h_shift, v_shift = np.random.randint(0, high=self.args.max_shift, size=2, dtype=int)

        # Shear
        if np.random.uniform() < self.args.shear_prob:
            shear = np.random.uniform(low=-self.args.max_shear, high=self.args.max_shear)

        for i in range(len(im_stack)):

            im_stack[i] = TF.affine(self.add_gaussian_noise(im_stack[i],  std=self.args.gauss_noise,
                                                            dimensions=self.dimensions),
                                    degrees, [h_shift, v_shift], zooming, shear,
                                    interpolation=transforms.InterpolationMode.BILINEAR,
                                    fill=im_stack[i].getpixel((5, 5))
                                    )

        return im_stack

    def __getitem__(self, index):

        # We should have different preprocessings for train and test.
        # We could perform data augmentations. For now we only leave a reminder.

        img_path = self.data.iloc[index]['path']
        im_stack = self.get_stack(img_path)  # list of PIL images

        if self.mode == 'train':
            im_stack = self.data_aug(im_stack)

        to_tensor = transforms.ToTensor()

        tensor_stack = [to_tensor(im) for im in im_stack.copy()]
        tensor_stack = torch.cat(tensor_stack)

        normalize = transforms.Normalize(
            mean=[ 0.2229,  0.2229,  0.2229],
            std=[  0.0969,  0.0969,  0.0969,])

        tensor_stack = normalize(tensor_stack)

        labels = self.data.iloc[index]['morgan_fp']
        labels = torch.from_numpy(labels)

        return tensor_stack, labels.type(torch.FloatTensor)

    def __len__(self):
        return len(self.data)


def parse_k1_to_k(path_k1, parsing_dict=None, K=1):
    '''Function for parsing from K1 folder to any K. By default
       K=1, making this function is the identity.'''

    if parsing_dict is None:
        parsing_dict = dict({
                    1   : '_K040_Amp040',
                    2   : '_K040_Amp060',
                    3   : '_K040_Amp080',
                    4   : '_K040_Amp100',
                    5   : '_K040_Amp120',
                    6   : '_K040_Amp140',
                    7   : '_K060_Amp040',
                    8   : '_K060_Amp060',
                    9   : '_K060_Amp080',
                    10  : '_K060_Amp100',
                    11  : '_K060_Amp120',
                    12  : '_K060_Amp140',
                    13  : '_K080_Amp040',
                    14  : '_K080_Amp060',
                    15  : '_K080_Amp080',
                    16  : '_K080_Amp100',
                    17  : '_K080_Amp120',
                    18  : '_K080_Amp140',
                    19  : '_K100_Amp040',
                    20  : '_K100_Amp060',
                    21  : '_K100_Amp080',
                    22  : '_K100_Amp100',
                    23  : '_K100_Amp120',
                    24  : '_K100_Amp140',
                    })

    assert K in parsing_dict.keys(), "This K folder doesn't exist or is not present in parsing_dict argument"

    params_str = parsing_dict[K]
    return path_k1.replace('/K-1/', f'/K-{K}/').replace('_K040_Amp040', params_str)



def dataset_ks_dict(dataset_k1_df, parsing_func):
    '''Create dictionary where keys are the K number corresponding to a set of
       operational parameters and the values are the pandas DataFrames pointing at images
       that folder

       ARGS:   dataset_k1_df: dataset where the directions point to K1
               parsing_func : function to parse the paths from K1 to any K folder
       OUTPUT: dataset_ks_dict
               '''
    dataset_list = list()

    for i in range(1,25):
        dataset_k = dataset_k1_df.copy()
        dataset_k['path'] = dataset_k['path'].apply(parse_k1_to_k, parsing_dict=None, K=i)
        dataset_list.append(dataset_k)

    return dict(zip(np.arange(1,25), dataset_list))



def parse_val_ks(val_df, k_list=None):
    '''Path parser for the validation dataframe. Converts from a DataFrame pointing to K1 samples
        to another one with even representation from the passed list of folders.

        ARGS:
                val_df: validation DataFrame containing paths pointing to K1
                k_list: list of K folder from where we want to evenly sample.
                        By default we use all the 24 folders.'''


    if k_list is None:
        k_list = np.arange(1,25)

    len_per_k = len(val_df)//len(k_list)
    val_k_df = val_df.copy()
    path_list = list(val_k_df['path'])
    for i in range(len(k_list)):
        K = k_list[i]
        for j in range(len_per_k):
            path_list[j+i*len_per_k] = parse_k1_to_k(path_list[j+i*len_per_k], parsing_dict=None, K=K)

    val_k_df['path'] = path_list
    return val_k_df
