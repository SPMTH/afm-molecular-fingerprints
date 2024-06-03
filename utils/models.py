# -*- coding: utf-8 -*-
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from torch import optim
from torchvision import datasets, transforms, models
from torchvision.models import resnet18, resnet50, ResNet50_Weights, efficientnet_b0, EfficientNet_B0_Weights
import torchvision.transforms.functional as TF

## 2D MODELS

def resnet_10_chan(output_size = 100):
    '''INPUT: output size
        OUTPUT: resnet-50 model with 10 channels in the first conv layer
                and a final linear layer with output size output_size'''

    resnet_10_chan = resnet50(weights=ResNet50_Weights.DEFAULT)
    weight = resnet_10_chan.conv1.weight.clone()

    resnet_10_chan.conv1 = nn.Conv2d(10, 64, kernel_size=7, stride=2, padding=3, bias=False)

    with torch.no_grad():
        resnet_10_chan.conv1.weight[:, 0:3] = weight
        resnet_10_chan.conv1.weight[:, 3:6] = weight
        resnet_10_chan.conv1.weight[:, 6:9] = weight
        resnet_10_chan.conv1.weight[:, 9] = weight[:,0]


    resnet_10_chan.fc = nn.Linear(2048, output_size, bias=True)
    return resnet_10_chan



def resnet_bias_warmer(model, train_df, device, eps = 1e-5):
    '''Calculates appropiate bias for the last linear layer of Resnet.
       Avoids hockeysticks losses.'''

    y = train_df['morgan_fp'].mean(axis = 0)
    x = np.log(y/(1-y) + eps)

    with torch.no_grad():
        for i in range(len(x)):
            model.fc.bias[i] = x[i]

    return model




def effnet_10_chan(output_size=100, dropout=None):
    '''INPUT:
        output_size: size of the final vector
        dropout: dropout probability. If None, dropout is set to 0 and instead
                batchnorm is used.


        OUTPUT: Efficient Net B0 model with 10 channels in the first conv layer
                and a final linear layer with output size output_size'''
    eff_net_10_chan = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    weight = eff_net_10_chan.features[0][0].weight.clone()

    eff_net_10_chan.features[0][0] = nn.Conv2d(10, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

    with torch.no_grad():
        eff_net_10_chan.features[0][0].weight[:, 0:3] = weight
        eff_net_10_chan.features[0][0].weight[:, 3:6] = weight
        eff_net_10_chan.features[0][0].weight[:, 6:9] = weight
        eff_net_10_chan.features[0][0].weight[:, 9] = weight[:, 0]
    eff_net_10_chan.classifier[1] = nn.Linear(in_features=1280, out_features=output_size, bias=True)

    if dropout is not None:
        eff_net_10_chan.features[0][1] = nn.Dropout2d(p=0.5)

    return eff_net_10_chan


def eff_net_bias_warmer(model, train_df, device, eps = 1e-5):
    '''Calculates appropiate bias for the last linear layer of EfficientNet.
       Avoid hockeysticks losses.'''

    y = train_df['morgan_fp'].mean(axis = 0)
    x = np.log(y/(1-y) + eps)

    with torch.no_grad():
        for i in range(len(x)):
            model.classifier[1].bias[i] = x[i]

    return model




## 3D MODELS


class ResNet3D(nn.Module):
    '''
    3D model based on a ResNet-18 backbone. The first part of the network (before layer1) has
    been changed to perform a 3D-convolution on the stack of images. Then, a regular 2D-convolution is used the reduce the
    number of channels. Dropout has been added to reduce co-adaptation.
    '''


    def __init__(self, output_size=1024):
        super(ResNet3D, self).__init__()

        # Load the pre-trained ResNet18 model to serve as backbone
        backbone = models.resnet18(pretrained=True)

        # We modify the first part of the network to use a 3D convolution.
        self.conv1 = nn.Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        self.conv1.weight.data = backbone.conv1.weight.unsqueeze(1)
        self.act = nn.ReLU()
        self.dropout3d = nn.Dropout3d(p=0.2)
        self.maxpool2D = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1)

        self.conv2 = nn.Conv2d(640, 64, 3, stride=1, padding=1)
        self.dropout2d = nn.Dropout2d(p=0.2)

        # Use the backbone
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.avgpool = backbone.avgpool
        self.fc = nn.Linear(512, output_size)

    def forward(self, x):

        # 3D plug
        x = x.unsqueeze(1)
        x = self.conv1(x)
        bs, [img_w, img_h] = x.size()[0], x.size()[3:5]
        x = self.act(x)
        x = self.dropout3d(x)
        x = x.reshape([bs, -1, img_w, img_h]) # reshape to squeeze
        x = self.maxpool2D(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.dropout2d(x)

        # Backbone
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class AtomCountPredictor(torch.nn.Module):
    '''Use a pretrained model as backbone to predict the atom counts
        of images. It outputs a vector with format
        ['C', 'Br', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S', 'H']

        Example: from image stack of molecule C22H13N3O4:
        output =  [22, 0, 0, 0, 0, 3, 4, 0, 0, 13]

        '''

    def __init__(self, pretrained_model, output_size=10):
        super(AtomCountPredictor, self).__init__()
        '''pretrained_model: expected to be EfficientNet model
           output_size: size of regression output vector'''

        self.backbone = pretrained_model.features
        self.avgpool = pretrained_model.avgpool
        self.regressor_head = nn.Sequential(
                            nn.Dropout(p=0.2, inplace=True),
                            nn.Linear(in_features=1280, out_features=output_size, bias=True),
                            nn.ReLU(inplace=False)
                          )

    def warm_bias(self, atom_counts):
        '''Adjust the bias of the final linear layer to have atom_counts as
            starting weights'''

        with torch.no_grad():
            for i in range(len(atom_counts)):
                self.regressor_head[1].bias[i] = atom_counts[i]


    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = x.squeeze()
        x = self.regressor_head(x)
        return x


def regressor_from_checkpoint(checkpoint_path, args, device):
    '''Creates regressor loading backbone from a morgan fingerprint model checkpoint'''

    pretrained_model = effnet_10_chan(output_size=args.n_fp, dropout=args.dropout)
    checkpoint = torch.load(checkpoint_path)
    pretrained_model.load_state_dict(checkpoint['model_state_dict'])

    return AtomCountPredictor(pretrained_model, output_size=args.output_size).to(device)
