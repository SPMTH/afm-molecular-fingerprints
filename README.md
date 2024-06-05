## Introduction
Non-Contact Atomic Force Microscopy with CO-functionalized metal tips (referred to as HR-AFM) provides access to the internal structure of individual molecules adsorbed on a surface with totally unprecedented resolution. Previous works have shown that deep learning (DL) models can retrieve the chemical and structural information encoded in a 3D stack of constant-height HR-AFM images, leading to molecular identification. In this work,  we overcome their limitations by using a well-established description of the molecular structure in terms of topological fingerprints, the 1024-bit Extended Connectivity Chemical Fingerprints of radius 2 (ECFP4), that were developed for substructure and similarity searching. ECFPs provide local structural information of the molecule, each bit correlating with a particular substructure within the molecule. Our DL model is able to extract this optimized structural descriptor from the 3D HR--AFM stacks and use it, through virtual screening, to identify molecules from their predicted ECFP4 with a retrieval accuracy on theoretical images of 95.4%. Furthermore, this approach,  unlike previous DL models, assigns a confidence score, the Tanimoto similarity, to each of the candidate molecules, thus providing information on the reliability of the identification.
 By construction, the number of times a certain substructure is present in the molecule is lost during the hashing process, necessary to make them useful for machine learning applications.  We show that it is possible to complement the fingerprint-based virtual screening with global information provided by another DL model that predicts from the same HR-AFM stacks the chemical formula, boosting the identification accuracy up to a 97.6%.  Finally, we perform a limited test with experimental images, obtaining promising results towards the application of this pipeline under real conditions.


 **Keywords**
 atomic force microscopy, molecular identification, chemical characterization, on surface synthesis, deep learning, neural networks, molecular fingerprints, density functional theory

This repository is the associated code for Molecular Identification via Molecular Fingerprint extraction from Atomic Force Microscopy images
 (https://arxiv.org/abs/2405.04321)

## Reproducing the figures
To reproduce some of the figures, you need to download the data and models folders from Zenodo.
 

