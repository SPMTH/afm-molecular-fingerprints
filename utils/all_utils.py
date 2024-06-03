# -*- coding: utf-8 -*-

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





def save_config(args):
    config_path = os.path.join(args.exp_path, 'config.txt')
    with open(config_path, 'w') as f:
        for key, value in args.__dict__.items():
            print(key, ':', value, file=f)


def train_for_epoch(model, train_loader, optimizer, criterion, device, args, epoch):

    # put the model in training mode
    model.train()


    n_batches = len(train_loader) // args.virtual_epochs


    train_losses = []
    if args.tqdm_flag:
        train_loader = tqdm(train_loader)

    for i, (batch, target) in enumerate(train_loader):

            # data to GPU
            batch = batch.to(device)
            target = target.to(device)

            # reset optimizer
            optimizer.zero_grad()

            # forward pass
            predictions = model(batch)

            # calculate loss
            loss = criterion(predictions, target)

            # backward pass
            loss.backward()

            # parameter update
            optimizer.step()

            # track loss
            train_losses.append(float(loss.item()))


            if ((i % n_batches)==0) and (i != 0) and (i != args.virtual_epochs*n_batches):
                print(f'epoch: {epoch}, virtual_epoch: { i // n_batches}')
                torch.save({'epoch': epoch,
                        'virtual_epoch': i // n_batches,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, os.path.join(args.models_path, f'checkpoint_{epoch}_virtual_epoch_{ i // n_batches}.pth'))

    train_losses = np.array(train_losses)
    train_loss = np.mean(train_losses)

    return train_loss

def train(optimizer, args, first_epoch=1,):

    '''Training loop'''


    train_losses , test_losses,   = [],  [],
    test_precisions, test_recalls, test_F1Scores = [], [], []

    for epoch in range(first_epoch, args.epochs+first_epoch):

        # train
        train_loss = train_for_epoch(model, train_loader, optimizer, criterion, device)

        # test
        test_loss, test_precision, test_recall, test_F1Score = validate(model, test_loader, criterion, device,
                                                                        metric_collection)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        #metrics
        test_precisions.append(test_precision)
        test_recalls.append(test_recall)
        test_F1Scores.append(test_F1Score)

         # print status
        print(f'[{epoch:02d}/{args.epochs:02d}] train loss: {train_loss:0.04f}  '
              f'valid loss: {test_loss:0.04f}  '
              f'valid precision: {test_precision:0.04f}  '
              f'valid recall: {test_recall:0.04f}  '
              f'valid F1 Score: {test_F1Score:0.04f}  '

             )



    return train_losses, test_losses, test_precisions, test_recalls, test_F1Scores



def validate(model, test_loader, criterion, device, metric_collection):
    model.eval()

    valid_losses = []


    with torch.no_grad():
        for batch, target in test_loader:

            # move data to the device
            batch = batch.to(device)
            target = target.to(device)

            # make predictions
            predictions = model(batch)

            # calculate loss
            loss = criterion(predictions, target)

            # calculate batch metrics
            _ = metric_collection(predictions, target)

            # track losses and predictions
            valid_losses.append(float(loss.item()))


    valid_losses = np.array(valid_losses)



    # calculate the mean validation loss
    valid_loss = np.mean(valid_losses)

    # calculate total metrics
    metrics = metric_collection.compute()

    valid_precision = metrics.get('MultilabelPrecision').cpu().numpy()

    valid_recall = metrics.get('MultilabelRecall').cpu().numpy()

    valid_F1Score = metrics.get('MultilabelF1Score').cpu().numpy()


    # reset valid metrics
    metric_collection.reset()

    return valid_loss, valid_precision, valid_recall, valid_F1Score


def train_test_split(dataset_df, seed, ratio ):
    '''Splits dataset in train and test with a ratio according
    to the parameter set in the config object

    INPUTS: dataset_df: the whole dataset

    OUTPUTS: train_df: the train split
             test_df: the test split'''




    np.random.seed(seed)
    n_samples = len(dataset_df)
    train_lim = int(ratio * n_samples)

    index_list = np.arange(n_samples)

    np.random.shuffle(index_list)


    train_index = index_list[:train_lim]
    test_index = index_list[train_lim:]

    train_df = dataset_df.iloc[train_index]
    test_df = dataset_df.iloc[test_index]

    return train_df, test_df



def compute_pos_weights(train_df, device, eps=1e-5):
    '''ARGS: train_df: DataFrame with fingerprints in numpy array
                with column name morgan_fp

        OUTPUT: pos_weight: torch tensor with '''

    pos = train_df['morgan_fp'].sum(axis = 0) + eps
    neg = len(train_df) - pos

    return torch.from_numpy(neg/pos).to(device)

def balance_beta_pos_weights(train_df, device, beta=1, eps=1e-5):


    '''
    Computes pos_weights but performs the operation p' = 1 + (p-1)/beta
    for p>1. This may help balancing precision and recall.

    ARGS: train_df: DataFrame with fingerprints in numpy array
                with column name morgan_fp

        OUTPUT: pos_weight: torch tensor with '''

    pos = train_df['morgan_fp'].sum(axis = 0) + eps
    neg = len(train_df) - pos
    pos_weights = neg/pos
    pos_weights[pos_weights>1] = 1 + (pos_weights[pos_weights>1] -1)/beta

    return torch.from_numpy(pos_weights).to(device)



def tanimoto_torch(fp1_t, fp2_t):
    a = torch.sum(fp1_t, dim=-1)
    b = torch.sum(fp2_t, dim=-1)
    c = torch.sum(fp1_t*fp2_t, dim = -1)
    return c/(a+b-c)


def validate_tanimoto(model, test_loader, criterion, device, metric_collection):
    model.eval()

    valid_losses = []
    valid_tanimoto = []


    with torch.no_grad():
        for batch, target in test_loader:

            # move data to the device
            batch = batch.to(device)
            target = target.to(device)

            # validate_tanimotomake predictions
            predictions = model(batch)

            # calculate loss
            loss = criterion(predictions, target)

            # calculate batch metrics
            _ = metric_collection(predictions, target)

            # calculate batch tanimoto similarity
            fp_pred = (torch.sigmoid(predictions) > 0.5)
            tan_batch = torch.mean(tanimoto_torch(fp_pred, target)).cpu().detach().numpy()
            valid_tanimoto.append(tan_batch)


            # track losses and predictions
            valid_losses.append(float(loss.item()))


    valid_losses = np.array(valid_losses)
    valid_tanimoto = np.array(valid_tanimoto)


    # calculate the mean validation loss
    valid_loss = np.mean(valid_losses)


    # calculate total metrics
    metrics = metric_collection.compute()

    valid_precision = metrics.get('MultilabelPrecision').cpu().numpy()

    valid_recall = metrics.get('MultilabelRecall').cpu().numpy()

    valid_F1Score = metrics.get('MultilabelF1Score').cpu().numpy()

    # calculate the mean validation tanimoto
    valid_tanimoto = np.mean(valid_tanimoto)


    # reset valid metrics
    metric_collection.reset()

    return valid_loss, valid_precision, valid_recall, valid_F1Score, valid_tanimoto


def validate_regression(model, test_loader, criterion, device):
    '''Validate regression model.

    OUTPUTS: valid_accuracy: here accuracy is defined as predicting
                             correctly the frequency of all atoms
             acc_atom_dict: accuracy of prediction per atom
        '''

    model.eval()

    valid_losses = []
    valid_accuracy = []
    valid_acc_atoms = []

    with torch.no_grad():
        for batch, target in test_loader:

            # move data to the device
            batch = batch.to(device)
            target = target.to(device)

            # make predictions
            predictions = model(batch)

            # calculate loss
            loss = criterion(predictions, target)

            # calculate accuracy
            mask = torch.eq(target, torch.round(predictions))
            accuracy = torch.mean(torch.all(mask, dim = 1).float()).cpu().detach().numpy()
            acc_atoms = torch.mean(mask.float(), dim=0).cpu().detach().numpy()



            # track losses and accuracy
            valid_losses.append(float(loss.item()))
            valid_accuracy.append(accuracy)
            valid_acc_atoms.append(acc_atoms)





    valid_losses = np.array(valid_losses)
    valid_accuracy = np.array(valid_accuracy)
    valid_acc_atoms = np.array(valid_acc_atoms)






    # calculate the mean validation loss
    valid_loss = np.mean(valid_losses)
    valid_accuracy = np.mean(valid_accuracy)
    valid_acc_atoms = valid_acc_atoms.mean(axis=0)

    # Create acc_atom_dict
    atom_labels = ['C', 'Br', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S', 'H']
    acc_atom_dict = dict(zip(atom_labels, valid_acc_atoms))


    return valid_loss, valid_accuracy, acc_atom_dict
