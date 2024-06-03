## Introduction
The identification of molecular structure and composition is often achieved through the application of spectroscopic techniques, such as vibrational spectroscopy methods, nuclear magnetic resonance (NMR) and mass spectrometry. These methods only give averaged information over macroscopic samples in disolution. In contrast, noncontact atomic force microscopy with CO--functionalized metal tips (referred as HR-AFM) provides insights into the internal structure of individual molecules on a surface.
From previous works, we know that 3D stacks of constant--height HR--AFM contain both structural and chemical information, as they allow for observation of the local
decay of the sample's charge density and the electrostatic field. In this work, we present a pipeline for complete identification (atomic species and topology) of quasi-planar organic molecules, where a Convolutional Neural network is trained to predict molecular fingerprints (ECFP4) from the 3D HR--AFM stack. The predicted fingerprints are later used to identify the molecule using a virtual screening procedure, providing a retrieval accuracy of 95.43\% on theoretical images. We benchmark our method against both theoretical and experimental images, with very promising results towards the application in the lab.

## Repository description

To run this code, create a conda environment and install the packages included in the requirements.txt file.


Below lies a brief description of the contents of this repository:

## Utils
In the utils folder we have all the model definitions, training and tracking functions, virtual screening, etc. It's basically the folder to copy to reproduce the core code in the repository.

## Models
Here we save the definitive molecular fingerprints and chemical formula model weights. When doing inference, don't forget to set model.eval() to freeze the batchnorm.

## Calculations
Scripts to obtain the data for some figures, like statistical analysis of virtual screening accuracy.

## Figures
Notebooks for reproducing the figures in the manuscript. I leave them there for inspiration and in case someone has to reuse them.

## Image analysis
This folder was a Proof of Concept for different interpolation strategies. 

## Parsing
Some useful scripts for creating the dataset of fingerprints and chemical formulas from SMILES strings and POSCARs of the simulation respectively.

## Testing
In principle, I wanted to write tests for my code. You can see how it went...

## training_scripts
Python training loops and submission scripts specific for the Rocinante machine.
