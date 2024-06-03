# -*- coding: utf-8 -*-
# IMPORTS
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
import json

# custom functions
from utils.all_utils import train_test_split, compute_pos_weights, train_for_epoch, save_config, validate, \
                            balance_beta_pos_weights, tanimoto_torch, validate_tanimoto

from utils.models import resnet_10_chan, effnet_10_chan, eff_net_bias_warmer

from utils.dataloader import  QUAM_with_noise, dataset_ks_dict, parse_k1_to_k, parse_val_ks





def virtual_epoch_paths(exp_path: str, return_dict=True):
    '''Extracts the paths of the models under the folder exp_path/models

    Example: for a model trained with 3 virtual epochs

        > vir_epoch_paths(exp_path='./experiments/300k_1024_all_ks')

        > ['./experiments/300k_1024_all_ks/models/checkpoint_1_virtual_epoch_1.pth',
        >   './experiments/300k_1024_all_ks/models/checkpoint_1_virtual_epoch_2.pth',
        >   './experiments/300k_1024_all_ks/models/checkpoint_1.pth',
        >   './experiments/300k_1024_all_ks/models/checkpoint_2_virtual_epoch_1.pth',
        >   './experiments/300k_1024_all_ks/models/checkpoint_2_virtual_epoch_2.pth',
        >   './experiments/300k_1024_all_ks/models/checkpoint_2.pth']
        '''


    models_path = os.path.join(exp_path, 'models')


    if return_dict:
        return {k:None for k in glob.glob(os.path.join(models_path, '*'))}

    else:
        return glob.glob(os.path.join(models_path, '*'))



def evaluate_virtual_epochs(exp_path, model, test_loader, criterion, device, metric_collection, print_flag=True,
                           save_flag=True):
    '''Function for evaluating the Morgan Fingerprints model on virtual epochs.
    INPUTS: exp_path: experiment path. The models exist under the folder exp_path/models.
            model: base model. We only change the weights for evaluating on different virtual epochs.
            test_loader: validation DataLoader.
            criterion: criterion to obtain validation loss.
            device: device on which the model is evaluated.
            metric_collection: torchmetrics MetricCollection object which contains the metrics
                                we need to calculate.
            print_flag: if True, prints the path of the model evaluating. It's useful to have a progress track.
            save_flag: if True, save the metrics JSON each iteration of the evaluation loop.

    OUTPUTS: metrics_dict: same input dictionary but the values are dictionaries containing loss and
                            the relevant metrics

    EXAMPLE:
    >  metrics_dict = virtual_epoch_paths(exp_path='./experiments/300k_1024_all_ks')

    >  print(evaluate_virtual_epochs(metrics_dict, model, test_loader, criterion, device, metric_collection))
    -------------------
    >  {'./experiments/300k_1024_all_ks/models/checkpoint_1_virtual_epoch_1.pth': {'test_loss': 0.17524171248078346,
    >    'precision': 0.44721755385398865,
    >    'recall': 0.4916732609272003,
    >    'F1_score': 0.4401499927043915,
    >    'tanimoto': 0.561638355255127},
    >   './experiments/300k_1024_all_ks/models/checkpoint_2_virtual_epoch_2.pth': {'test_loss': 0.1107240617275238,
    >    'precision': 0.6171062588691711,
    >    'recall': 0.6556202173233032,
    >    'F1_score': 0.6128804087638855,
    >    'tanimoto': 0.7562735080718994},
    >   './experiments/300k_1024_all_ks/models/checkpoint_2_virtual_epoch_3.pth': {'test_loss': 0.10860804654657841,
    >    'precision': 0.6080066561698914,
    >    'recall': 0.6642138957977295,
    >    'F1_score': 0.6097334027290344,
    >    'tanimoto': 0.7554464936256409},
    >   './experiments/300k_1024_all_ks/models/checkpoint_4.pth': {'test_loss': 0.11208268068730831,
    >    'precision': 0.6546374559402466,
    >    'recall': 0.6799007654190063,
    >    'F1_score': 0.6453921794891357,
    >    'tanimoto': 0.7944305539131165}}

            '''
    metrics_dict = virtual_epoch_paths(exp_path)

    for models_path in metrics_dict.keys():
        if print_flag:
            print(f'Evaluating {models_path}')

        checkpoint = torch.load(models_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()

        test_loss, test_precision, test_recall, test_F1Score, test_tanimoto = \
                    validate_tanimoto(model, test_loader, criterion, device, metric_collection)

        metrics_dict[models_path] =  {  'test_loss': float(test_loss),
                                    'precision' : float(test_precision.tolist()),
                                    'recall'    : float(test_recall.tolist()),
                                    'F1_score'  : float(test_F1Score.tolist()),
                                    'tanimoto'  : float(test_tanimoto)
                                                        }
        if save_flag:
            file_path = os.path.join(exp_path, 'metrics/virtual_epochs.json')
            with open(file_path, 'w') as fp:
                json.dump(metrics_dict, fp, indent=4)


    return metrics_dict



def eval_model_on_k_folders(models_path, model, val_dict, criterion, device, metric_collection, args,
                            print_flag=True, save_flag=True):

    '''Function for evaluating a model over several folders.
    INPUTS: models_path: path where the model is located. You need to point to an specific (virtual) epoch.
            model: base model. We only change the weights for evaluating on different virtual epochs.
            val_dict: dictionary where keys are the index of the K folder and the values are the datasets of
                        paths pointing to that folder.
            criterion: criterion to obtain validation loss.
            device: device on which the model is evaluated.
            metric_collection: torchmetrics MetricCollection object which contains the metrics
                                we need to calculate.
            args: args object containing the batch_size hyperparameter.
            print_flag: if True, prints the path of the model evaluating. It's useful to have a progress track.
            save_flag: if True, save the metrics JSON each iteration of the evaluation loop.

    OUTPUTS: k_metrics_dict: dictionary where keys are the index of K folders and values
                are dictionaries containing loss and the relevant metrics.

    EXAMPLE:

    > models_path = '/home/manuel/CODE/Pytorch-fingerprints/experiments/300k_1024_all_ks \
                        /models/checkpoint_1_virtual_epoch_1.pth'

    > print(eval_model_on_k_folders(models_path, model, val_dict,
                    criterion, device, metric_collection, args, print_flag=True, save_flag=True))
    ---------------------------------
    {
    "1": {
        "test_loss": 0.17524171248078346,
        "precision": 0.44721755385398865,
        "recall": 0.4916732609272003,
        "F1_score": 0.4401499927043915,
        "tanimoto": 0.561638355255127
    },
    "2": {
        "test_loss": 0.17289534956216812,
        "precision": 0.44704294204711914,
        "recall": 0.4947711229324341,
        "F1_score": 0.44213563203811646,
        "tanimoto": 0.5680942535400391
    }
    }

    '''



    checkpoint = torch.load(models_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()


    k_metrics_dict = {int(k): None for k in val_dict.keys()}

    for k_index in k_metrics_dict.keys():
        if print_flag:
            print(f'Evaluating from folder K-{k_index}')

        testset = QUAM_with_noise(args, val_dict[k_index], mode='test')
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=6,
                                              pin_memory=True)

        test_loss, test_precision, test_recall, test_F1Score, test_tanimoto = \
                        validate_tanimoto(model, test_loader, criterion, device, metric_collection)

        k_metrics_dict[k_index] =  {  'test_loss': float(test_loss),
                                      'precision' : float(test_precision.tolist()),
                                      'recall'    : float(test_recall.tolist()),
                                      'F1_score'  : float(test_F1Score.tolist()),
                                      'tanimoto'  : float(test_tanimoto)
                                                        }

        exp_path = os.path.join('/',*models_path.split('/')[:-2])
        if save_flag:
            file_path = os.path.join(exp_path, 'metrics', 'k_metrics_'+models_path.split('/')[-1]+'.json')
            with open(file_path, 'w') as fp:
                json.dump(k_metrics_dict, fp, indent=4)
    return k_metrics_dict
