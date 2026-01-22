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
based on mean-color distances, our framework employs a distributional OT distance, yielding a mathematically unified formulation across both clustering levels.
## Installation
This project can directly run in Matlab without any additional installation because we have compiled third part C++ (in algorithm folder) into MATLAB file or EXE file.  However, if you want to adapt the code for your own use, you can compile the c++ code by yourself. 
## Usage
We provide two examples segmentation from our paper. Please find them in code folder. In particular, the paths are code\example1\seg_by_our_AR_ZZ.mlx and code\example2\seg_by_our_ar_zz.mlx


