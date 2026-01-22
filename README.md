# Project Title
Superpixel-Based Image Segmentation Using Squared 2-Wasserstein Distances
## Description
We present an efficient method for image segmentation in the presence of strong
inhomogeneities. The approach can be interpreted as a two-level clustering
procedure: pixels are first grouped into superpixels via a linear least-squares
assignment problem, which can be viewed as a special case of a discrete optimal
transport (OT) problem, and these superpixels are subsequently greedily merged
into object-level segments using the squared 2-Wasserstein distance between their
empirical distributions. In contrast to conventional superpixel merging strategies
based on mean-colour distances, our framework employs a distributional OT distance, yielding a mathematically unified formulation across both clustering levels.
## Installation
This project can run directly in MATLAB without any additional installation, since we have compiled third-party C++ code (in the algorithm folder) into a MATLAB file or an EXE.  However, if you want to adapt the code for your own use, you can compile the C++ code by yourself. 
## Usage
We provide two segmentation examples from our paper. Please find them in the code folder. In particular, the paths are code\example1\seg_by_our_AR_ZZ.mlx and code\example2\seg_by_our_ar_zz.mlx


