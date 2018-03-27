# Weakly-Supervised Convolutional Neural Networks for Multimodal Image Registration

## Introduction
This is a tutorial aiming to use concise and self-explanatory code to describe the implementation of the deep-learning-based image registration method in [Hu et al 2018][Hu2018a] (and the preliminary work was published in [Hu et al ISBI2018][Hu2018b]). A full re-implementation is available at [NiftyNet platform][niftynet]. The sections are organised as follows:

* [1 - Multimodal Image Registration](#section1)
* [2 - Example Data](#section2)
* [3 - Weakly-Supervised Dense Correspondence Learning](#section3)
* [4 - Label Similarity Measures](#section4)
* [5 - Training](#section5)
* [6 - Deformation Regularisation](#section6)
* [7 - Convolutional Neural Networks for Predicting Displacements](#section7)
* [8 - Weakly-Supervised Registration Revisted](#section8)

[Hu2018a]: https://arxiv.org/abs/1711.01666
[Hu2018b]: 
[niftynet]: http://niftynet.io/


## <a name="section1"></a>1 - Multimodal Image Registration
Medical image registration aims to find a dense displacemnt field (DDF), such that a given "moving image" can be warped (resampled using the DDF) to match a second "fixed image", that is, the corresponding anatomical structures are in the same spatial location.

The definition of "multimodal" varies from changing of imaging parameters (such as MR sequancing or contrast agents) to different scanners. 
<p style=\"float: left; width: 85%; margin-right: 1%;\"><img src=\"./media/volume_mr.jpg\" /></p>\n
<p style=\"float: left; width: 85%; margin-right: 1%;\"><img src=\"./media/volume_us.jpg\" /></p>\n

## <a name="section2"></a>2 - Example Data
Some example data can be downloaded directly [here][data].

Unzip the data

[data]: 


## <a name="section3"></a>3 - Weakly-Supervised Dense Correspondence Learning
The idea of the wearkly-supervised learning is to use expert labels that represent the same anatomical structures. The label-driven method may be considered as a "lazy" method (e.g. compared to simulating complex biological deformation or engineering sophisticated similarity measure, used in supervised or unsupervised approches, respectively) or as a "industrious" method that requires a great amount manually-annotated anatomical structures in volumetric data, depending on one's viewpoint.

While the target is predicting DDF which we do not have ground-truth data, the method is considered as "weakly-supervised" because the anatomical labels are used only in training so at inference time, the registration does not need any labels (i.e. fully-automated image registration accepts a pair of images and predicts a DDF, without segmentation of kind to aid the alignment or even initialisation). They are treated as if they are the "target labels" instead of "input predictors" in a classical regression analysis. A more detailed discussion of the formulation of the "weakly-supervised registration" is provided in the later section and it is not in the papers! ;)

The trick here is to use ONLY images as input to the neural network without labels, but the netowrk-predicted DDF can be used to transform the associated labels (from the same images) to match to each other, as shown in the picture:


The main problems with label-driven registration methods are labels representing corresponding structures are inherently sparse - among training cases, the same anatomical structures are not always present between a given moving and fixed image pair for training; when available, they neither cover the entire image domain nor detailed voxel correspondence.


## <a name="section4"></a>4 - Label Similarity Measures




