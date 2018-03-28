# Weakly-Supervised Convolutional Neural Networks for Multimodal Image Registration


## Introduction
This is a tutorial aiming to use minimum and mostly self-explanatory scripts to describe the implementation of the deep-learning-based image registration method in [Hu et al 2018][Hu2018a] (and the preliminary work was published in [Hu et al ISBI2018][Hu2018b]). A full re-implementation with many other unitilities is available at [NiftyNet platform][niftynet]. The sections are organised as follows:

* [1. Multimodal Image Registration](#section1)
* [  ..* Example Data](#section1-1)
* [2. Weakly-Supervised Dense Correspondence Learning](#section2)
* [  ..* Label Similarity Measures](#section2-1)
* [  ..* Training](#section2-2)
* [  ..* Deformation Regularisation](#section2-3)
* [  ..* Convolutional Neural Networks for Predicting Displacements](#section2-4)
* [3. Try with Your Own Image-Label Data](#section3)
* [4. Weakly-Supervised Registration Revisted](#section4)


## <a name="section1"></a>1 Multimodal Image Registration
Medical image registration aims to find a dense displacemnt field (DDF), such that a given "moving image" can be warped (transformed using the predicted DDF) to match a second "fixed image", that is, the corresponding anatomical structures are in the same spatial location.

The definition of "multimodal" varies from changing of imaging parameters (such as MR sequancing or contrast agents) to images from different scanners. An example application here is to support 3D-ultrasound-guided intervention for prostate cancer patients. The ultrasound image looks like this:
<p style=\"float: left; width: 85%; margin-right: 1%;\"><img src=\"https://github.com/YipengHu/example-data/raw/master/label-reg-demo/media/volume_us.jpg\" /></p>
In these procedures, the urologists woulkd like to know where the supicious regions are, so they can take a tissue sample to confirm pathology (i.e. biopsy), they may want to treat the small areas such as a tumour (i.e. focal therapy). They also want to avoid other healthy delicate srrounding or internal structures such as nerves, urethra, rectum, bladder. However, from ultrasound imaging, it is already difficult to work out where the prostate gland is, let alone the detailed anatomical and pathlogical structures. 

MR imaging, on the other hand, provides a better tissue contrast to localise these interesting structures. Example slices of the 3D MR volume are here:
<p style=\"float: left; width: 85%; margin-right: 1%;\"><img src=\"https://github.com/YipengHu/example-data/raw/master/label-reg-demo/media/volume_mr.jpg\" /></p>
The catch is that the MR imaging is difficult and expensive to use _during_ those procedures, and requires more imaging time (you might need to wait half hour to get full multi-parametric sequences to reliably find where the tumour is), so they are usually available only _before_ the procedure. If the MR image could be aligned with the ultrasound image in real-time, all the information can then be "registered" from the MR to the ultrasound images. It sounds like a good solution. It turns out to be a very difficult problem, stimulating a decade-long research topic. The [journal articule][Hu2018b] provides many references and examples describing the detailed difficulties and attempted solutions. This tutorial describes an alternative using deep learning method. The main advantage is the resulting registration between the 3D data is fast (several 3D registrations in one second), fully-automated and easy to implement.


### <a name="section1-1"></a>Example Data
Due to medical data restrictions, we use some fake (fewer and smaller) data in this tutorial to mimic those from the prostate imaging application. You can download these by clicking the following link:
[**Example data**][data]
First you need to unzip the downloaded data to folders, which you need to specify in [config.py][config_file] in order to run the code.

In summary, for each numbered patient, there is a quartet of data, a 3D MR volume, a 3D ultrasound volume, several landmarks delineated from MR and ultrasound volumes, the latter two being in 4D binary volumes. The fourth dimention indicates different types of landmarks, such as the prostate gland and the apex/base points (where urethra enters and exists prostate gland).


## <a name="section2"></a>2 Weakly-Supervised Dense Correspondence Learning
The idea of the wearkly-supervised learning is to use expert labels that represent the same anatomical structures. Depending on one's personal viewpoint, this type of label-driven methods may be considered as being "lazy" (e.g. compared to simulating complex biological deformation or engineering sophisticated similarity measure, used in supervised or unsupervised approches, respectively) or being "industrious" as a great amount manually-annotated anatomical structures in volumetric data are requried.

While the goal is predicting DDF which we do not have ground-truth data for, the method is considered as "weakly-supervised" because the anatomical labels are used only in training so, at inference time, the registration does not need any labels (i.e. fully-automated image registration accepts a pair of images and predicts a DDF, without segmentation of any kind to aid the alignment or even initialisation). They are treated as if they are the "target labels" instead of "input predictors" in a classical regression analysis. Various formulations of the [weakly-supervised registration](#section9) is discussed and it is not in the papers! ;)

The trick here is to use ONLY images as input to the neural network without labels, but the netowrk-predicted DDF can be used to transform the associated labels (from the same images) to match to each other, as shown in the picture:
<p style=\"float: left; width: 85%; margin-right: 1%;\"><img src=\"https://github.com/YipengHu/example-data/raw/master/label-reg-demo/media/training.jpg" /></p>
<p style=\"float: left; width: 85%; margin-right: 1%;\"><img src=\"https://github.com/YipengHu/example-data/raw/master/label-reg-demo/media/inference.jpg" /></p>

The main problems with label-driven registration methods are labels representing corresponding structures are inherently sparse - among training cases, the same anatomical structures are not always present between a given moving and fixed image pair for training; when available, they neither cover the entire image domain nor detailed voxel correspondence. We solve the 


### <a name="section2-1"></a>Label Similarity Measures
Using cross-entropy to direct measure the loss between two given binary masks (representing the segmentation of the corresponding anatomies) has several problems:
1 - 

## <a name="section3"></a>3 Try with Your Own Image-Label Data
Anything readable by [NiBabel][nibabel] should be working with the DataReader

[data]: https://github.com/YipengHu/example-data/raw/master/label-reg-demo/data.zip
[config_file]: ./config.py
[Hu2018a]: https://arxiv.org/abs/1711.01666
[Hu2018b]: https://arxiv.org/abs/1711.01666
[niftynet]: http://niftynet.io/
[nibabel]: http://nipy.org/nibabel/



