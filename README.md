## Introduction
Non-Contact Atomic Force Microscopy with CO-functionalized metal tips (referred to as HR-AFM) provides access to the internal structure of individual molecules adsorbed on a surface with totally unprecedented resolution. Previous works have shown that deep learning (DL) models can retrieve the chemical and structural information encoded in a 3D stack of constant-height HR-AFM images, leading to molecular identification. In this work,  we overcome their limitations by using a well-established description of the molecular structure in terms of topological fingerprints, the 1024-bit Extended Connectivity Chemical Fingerprints of radius 2 (ECFP4), that were developed for substructure and similarity searching. ECFPs provide local structural information of the molecule, each bit correlating with a particular substructure within the molecule. Our DL model is able to extract this optimized structural descriptor from the 3D HR--AFM stacks and use it, through virtual screening, to identify molecules from their predicted ECFP4 with a retrieval accuracy on theoretical images of 95.4%. Furthermore, this approach,  unlike previous DL models, assigns a confidence score, the Tanimoto similarity, to each of the candidate molecules, thus providing information on the reliability of the identification.
 By construction, the number of times a certain substructure is present in the molecule is lost during the hashing process, necessary to make them useful for machine learning applications.  We show that it is possible to complement the fingerprint-based virtual screening with global information provided by another DL model that predicts from the same HR-AFM stacks the chemical formula, boosting the identification accuracy up to a 97.6%.  Finally, we perform a limited test with experimental images, obtaining promising results towards the application of this pipeline under real conditions.


 **Keywords**
 atomic force microscopy, molecular identification, chemical characterization, on surface synthesis, deep learning, neural networks, molecular fingerprints, density functional theory

This repository is the associated code for Molecular Identification via Molecular Fingerprint extraction from Atomic Force Microscopy images
 (https://arxiv.org/abs/2405.04321)

## Download from Zenodo
Since data and models folders exceed github file size limit, they are stored in Zenodo (https://zenodo.org/records/11483708).

To download them, just run:
```
chmod +x zenodo_download.sh
./zenodo_download.sh
```

The script downloads, untars and finally remove the tar files from your repository folder.

## Install the environment
You can install the environment via conda:
```
conda create --name <env_name> --file requirements.txt
```
Or pip:
```
pip install -r requirements_pip.txt
```

## QUAM-AFM dataset
To train your own models, you need to download the QUAM-AFM dataset in your machine. 
The dataset is freely available at https://edatos.consorciomadrono.es/dataset.xhtml?persistentId=doi:10.21950/UTGMZ7
This project assumes you are using an HPC environment where the QUAM-AFM is stored at /scratch/dataset/quam 

## Usage

Extracting the fingerprints is as easy as loading the checkpoint of the model you want to use:
```python
import pandas as pd
from utils.models import effnet_10_chan
from utils.screening import fp_screening_function predict_fp

model = effnet_10_chan(output_size=args.n_fp, dropout=args.dropout)

models_path = 'path/to/models/300k_1024_all_ks_dropout_0_5/models'
checkpoint = torch.load(os.path.join(models_path, 'checkpoint_5_virtual_epoch_7.pth'), map_location=torch.device('cpu'))

model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()
```
Loading the experimental images and performing inference on them:
```python
fp_pred = predict_fp(model, exp_molec_path)
```
It is important that the images of the molecule are ordered from 0 to 9, where 0 is the closest tip-sample image.


After the fingerprints are extracted, you can perform the virtual screening against your precomputed database of fingerprints to extract the top 5 candidates:
```python
data_path = 'path/to/data/dataset/285k_train_15k_fp_and_atom_counts_w_H.gz'
dataset_df = pd.read_pickle(data_path)

test_df = dataset_df[dataset_df['split'] == 'test']
test_df = parse_val_ks(test_df)

output_df = fp_screening_function(fp_pred, test_df, top_k = 5, int_type=np.int8)

