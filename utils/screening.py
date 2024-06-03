# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from torch import optim
from torchvision import datasets, transforms, models
import torchvision.transforms.functional as TF


def fp_screening_function(query, reference_df, top_k = None, int_type=None):
    '''Function to perform the screening of chemical fingerprints with respect
    to a pre-computed database, outputing the top k candidates in descending order.

    INPUTS:
            query: predicted morgan fingerprints to compare
            reference_df: pandas DataFrame containing the pre-computed
                        (CID, morgan fingerprint) pairs#
            top_k: number k of molecules to output. If None, the whole dataset is output'''

    #Preprocess data

    reference_arr = np.stack(reference_df['morgan_fp'].to_numpy())
    fp_size = reference_arr.shape[1]
    query = np.reshape(query, [1, fp_size])
    if int_type is not None:
        reference_arr = reference_arr.astype(int_type)
        query = query.astype(int_type)



    #Compute tanimoto similarity
    a = np.sum(query, axis = -1)
    b = np.sum(reference_arr, axis = -1)
    c = np.sum(reference_arr*query, axis=-1)
    tan = c/(a+b-c)

    #sort by tanimoto similarity
    output_df = reference_df.copy()
    output_df['tanimoto']= tan
    output_df = output_df.sort_values(by=['tanimoto'], ascending=False)

    if top_k is not None:
        output_df = output_df.iloc[:top_k]
    return output_df





def load_img_stack(path, dimensions = [224,224], zoom=1):
    '''Obtains stack of images from folder path and preprocess it so we can directly feed it to our CNN

        INPUT: path of the folder containing the AFM images of the molecule

        OUTPUT: list of PIL images'''

    last_str = path.split('/')[-1]
    im_list = list()
    for i in range(10):
        im_path = path + '/' + last_str + f'_df_00{i}.jpg'
        im_list.append(Image.open(im_path).resize(dimensions, resample=Image.Resampling.BILINEAR))

    # Default parameters
    degrees = 0
    h_shift = 0
    v_shift = 0
    shear = 0

    for i in range(len(im_list)):
        im_list[i] = TF.affine(im_list[i], degrees, [h_shift, v_shift], zoom, shear,
                            interpolation=transforms.InterpolationMode.BILINEAR,
                            fill=im_list[i].getpixel((5, 5))
                            )

    to_tensor = transforms.ToTensor()

    tensor_stack = [to_tensor(im) for im in im_list.copy()]
    tensor_stack = torch.cat(tensor_stack)

   #normalize = transforms.Normalize(
    #    mean=[0.1321, 0.1487, 0.1689, 0.1983, 0.2229, 0.2591, 0.3430, 0.4580, 0.5787, 0.6887],
    #    std=[0.0853, 0.0853, 0.0883, 0.0942, 0.0969, 0.1066, 0.1419, 0.1840, 0.2144, 0.2215]) #this is for K-1



    normalize = transforms.Normalize(
        mean=[0.2855, 0.3973, 0.4424, 0.4134, 0.6759, 1.0664, 0.9904, 0.7708, 0.5748, 0.4055],
        std=[1.1341, 1.2528, 1.3125, 1.3561, 1.5844, 1.7763, 1.5447, 1.2683, 1.0588, 0.9308]) # for all K folders

    tensor_stack = normalize(tensor_stack)

    return tensor_stack


def predict_fp(model, path, device, thres = 0.5):
    tensor_stack = load_img_stack(path)
    with torch.no_grad():
        batched_tensor = tensor_stack.unsqueeze(dim = 0).to(device)
        predictions = model(batched_tensor)
        fp_pred = (torch.sigmoid(predictions) > thres)

    return fp_pred.cpu().numpy().astype(int)

def tanimoto_numpy(fp1, fp2):
    a = np.sum(fp1, axis = -1)
    b = np.sum(fp2, axis = -1)
    c = np.sum(fp1*fp2, axis = -1)
    return c/(a+b-c)

def count_ties(output_df):
    '''Checks if there is a tie. If there is a tie, the output is
    the number of molecules in the tie. If not, output is 1.'''

    return len(output_df[ output_df['tanimoto'] == output_df['tanimoto'].iloc[0]])

def order_by_diff(output_df, pred_atom_count, n_ties, verbose=False):
    '''Reorders the output_df DataFrame in case of tie'''
    atom_count_ties = output_df.iloc[:n_ties][['C', 'Br', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S', 'H']].to_numpy()
    diff = ((pred_atom_count - atom_count_ties)**2).sum(axis=1)
    output_df_ordered = output_df.iloc[np.argsort(diff)]
    if verbose:
        print(f'diff: {diff}, order: {np.argsort(diff)}')
    return output_df_ordered
