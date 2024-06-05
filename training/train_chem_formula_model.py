# -*- coding: utf-8 -*-
# IMPORTS
import sys
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

USER = os.getenv('USER')
sys.path.append(f'/home/{USER}/CODE/Pytorch-fingerprints/')
# torch packages
import torch
import torchvision
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from torch import optim
from torchvision import datasets, transforms, models
from torchvision.models import resnet50, ResNet50_Weights, efficientnet_b0, EfficientNet_B0_Weights
import torchvision.transforms.functional as TF

from torchmetrics import MetricCollection
from torchmetrics.classification import MultilabelRecall, MultilabelPrecision, MultilabelF1Score

import torch.nn.functional as F
from torch.utils.data import Dataset

print('PyTorch version:', torch.__version__)



# custom functions
from utils.all_utils import train_for_epoch, save_config, validate_regression

from utils.models import resnet_10_chan, effnet_10_chan, regressor_from_checkpoint

from utils.dataloader import  QUAM_atom_regressor_10_imgs, dataset_ks_dict, parse_k1_to_k, parse_val_ks

print('Packages loaded')

# make sure to enable GPU acceleration!
print(f'available devices: {torch.cuda.device_count()}')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.device(0)
print(device, torch.cuda.get_device_name(0))


class configuration:
    def __init__(self):
        # EXPERIMENT PARAMETERS
        self.experiment_name = 'regression_w_H_lr_5e-4'
        self.n_fp = 1024  # number of fingerprints of the backbone model
        self.output_size = 10 # output size of regressor model
        self.ratio = 0.95  # train/test ratio
        self.seed = 42
        self.virtual_epochs = 5 # number of times we save the model per epoch. If equal to 1
                                # we only save at the end of each epoch.
        self.tqdm_flag = True

        # TRAINING PARAMETERS

        self.lr = 0.0005  # learning rate
        self.dropout = 0.5
        # self.momentum = 0.9  # momentum of SGD optimizer
        self.weight_decay = 0  # L2 regularization constant
        self.batch_size = 50  # Training batch size
        self.test_batch_size = 100  # Test batch size
        self.epochs = 30  # Number of epochs
        self.bias_warmer = True # setting appropiate bias
        self.pos_weight_balancer = True #for bigger fingerprints, it helps balance precision and recall
        self.pos_weight_beta = 10
        # DATA AUGMENTATION PARAMETERS

        # Rotation
        self.rot_prob = 0.5  # prob of rotation in data augmentation
        self.max_deg = 180  # maximum degrees of rotation in data augmentation

        # Zoom
        self.zoom_prob = 0.7  # prob of applying zoom
        self.max_zoom = 0.3  # maximum zooming in/out

        # Translation
        self.shift_prob = 0.3  # probability of vertical or/and horizontal translation
        self.max_shift = 20  # translation

        # Shear
        self.shear_prob = 0.3  # probability of shearing
        self.max_shear = 10  # maximum shearing angle

        # Gaussian noise
        self.gauss_noise = 2 # std of gaussian noise

        # comments
        self.comments = 'It uses all K folders'

        # METRICS AND MODELS PATHS
        self.exp_path = os.path.join(f'/home/{USER}/CODE/Pytorch-fingerprints/experiments',
            self.experiment_name)
        self.metrics_path = os.path.join(self.exp_path, 'metrics')
        self.models_path = os.path.join(self.exp_path, 'models')



## Create arguments object
args = configuration()
# Print experiment name
print('Experiment name:', args.experiment_name)
# Set random seed for reproducibility
torch.backends.cudnn.deterministic = True  # fix the GPU to deterministic mode
torch.manual_seed(args.seed)  # CPU seed
torch.cuda.manual_seed_all(args.seed)  # GPU seed
random.seed(args.seed)  # python seed for image transformation
np.random.seed(args.seed)


#Load data

data_path = f'/home/{USER}/QUAM-AFM/datasets/atoms_count_w_H_df.gz'
dataset_df = pd.read_pickle(data_path)


train_df = dataset_df[dataset_df['split'] == 'train']
val_df = dataset_df[dataset_df['split'] == 'val']
train_dict = dataset_ks_dict(train_df, parse_k1_to_k)

print('training dict created')

train_k_df = pd.concat(train_dict.values(), ignore_index=True)
val_k_df = parse_val_ks(val_df)

trainset = QUAM_atom_regressor_10_imgs(args, train_k_df, mode='train')
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=12,
                                           pin_memory=True)

testset = QUAM_atom_regressor_10_imgs(args, val_k_df, mode='test')
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=12,
                                           pin_memory=True)

print('Test set samples:', len(val_k_df))



model = effnet_10_chan(output_size=args.n_fp, dropout=args.dropout)
models_path = f'/home/{USER}/CODE/Pytorch-fingerprints/experiments/300k_1024_all_ks_dropout_0_5/models'
checkpoint_path = os.path.join(models_path, 'checkpoint_5_virtual_epoch_7.pth')
model = regressor_from_checkpoint(checkpoint_path, args, device)

if args.bias_warmer:
    atoms_mean = np.mean(train_df[['C', 'Br', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S', 'H']], axis = 0)
    model.warm_bias(atoms_mean)
    print('Bias warmed')

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

if not os.path.isdir(args.exp_path):
    os.makedirs(args.exp_path)
    os.makedirs(args.metrics_path)
    os.makedirs(args.models_path)

save_config(args)

train_losses, test_losses, = [], [],

test_accuracy, acc_C, acc_Br = [], [], []
acc_Cl, acc_F, acc_I, acc_N, acc_O, acc_P, acc_S, acc_H = [], [], [], [], [], [], [], []

print('Start training')

for epoch in range(1, args.epochs + 1):

    # train
    train_loss = train_for_epoch(model, train_loader, optimizer, criterion, device, args, epoch)

    # test
    valid_loss, valid_accuracy, acc_atom_dict = validate_regression(model, test_loader, criterion, device)


    train_losses.append(train_loss)
    test_losses.append(valid_loss)
    test_accuracy.append(valid_accuracy)

    acc_C.append(acc_atom_dict['C'])
    acc_Br.append(acc_atom_dict['Br'])
    acc_Cl.append(acc_atom_dict['Cl'])
    acc_F.append(acc_atom_dict['F'])
    acc_I.append(acc_atom_dict['I'])
    acc_N.append(acc_atom_dict['N'])
    acc_O.append(acc_atom_dict['O'])
    acc_P.append(acc_atom_dict['P'])
    acc_S.append(acc_atom_dict['S'])
    acc_H.append(acc_atom_dict['H'])

    # print status
    print(f'[{epoch:02d}/{args.epochs:02d}] train loss: {train_loss:0.04f}  '
          f'valid loss: {valid_loss:0.04f}  '
          f'valid accuracy: {valid_accuracy:0.04f}  '
          f'atom acc: {acc_atom_dict}'
          )

    # save losses
    np.save(os.path.join(args.metrics_path, 'LOSS_epoch_train.npy'), np.asarray(train_losses))
    np.save(os.path.join(args.metrics_path, 'LOSS_epoch_test.npy'), np.asarray(test_losses))
    np.save(os.path.join(args.metrics_path, 'Acc_epoch_test.npy'), np.asarray(test_accuracy))



    # save model on every epoch:

    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'valid_accuracy': valid_accuracy,
                'acc_atom_dict': acc_atom_dict,

                }, os.path.join(args.models_path, f'checkpoint_{epoch}.pth'))
