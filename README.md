# Weakly-Supervised Convolutional Neural Networks for Multimodal Image Registration


## Introduction
This tutorial aims to use minimum and self-explanatory scripts to re-work the implementation of a deep-learning-based image registration method in [Hu et al 2018][Hu2018a] (and the preliminary work was in [Hu et al ISBI2018][Hu2018b]). An efficient re-implementation with many other utilities is available at [NiftyNet Platform][niftynet]. The sections are organised as follows:

* [**1. Multimodal Image Registration**](#section1)
* [     - Example Data](#section1-1)
* [**2. Weakly-Supervised Dense Correspondence Learning**](#section2)
* [     - Label Similarity Measures](#section2-1)
* [     - Convolutional Neural Networks for Predicting Displacements](#section2-2)
* [     - Deformation Regularisation](#section2-3)
* [     - Training](#section2-4)
* [     - Inference](#section2-5)
* [**3. Try with Your Own Image-Label Data**](#section3)
* [**4. Weakly-Supervised Registration Revisted**](#section4)


## <a name="section1"></a>1 Multimodal Image Registration
Medical image registration aims to find a dense displacemnt field (DDF), such that a given "moving image" can be warped (transformed using the predicted DDF) to match a second "fixed image", that is, the corresponding anatomical structures are in the same spatial location. 

The definition of "multimodal" varies from changing of imaging parameters (such as MR sequancing or contrast agents) to images from different scanners. An example application here is to support 3D-ultrasound-guided intervention for prostate cancer patients. The ultrasound image looks like this:
![alt text](https://github.com/YipengHu/example-data/raw/master/label-reg-demo/media/volume_us.jpg "Ultrasound Image Volume")
In these procedures, the urologists woulkd like to know where the suspicious regions are, so they can take a tissue sample to confirm pathology, i.e. biopsy, or they may treat certain small areas such as a tumour, i.e. focal therapy. They also want to avoid other healthy delicate srrounding or internal structures such as nerves, urethra, rectum and bladder. However, from ultrasound imaging, it is hard enough to work out the boundaries of the prostate gland, let alone the detailed anatomical and pathlogical structures. 

MR imaging, on the other hand, provides a better tissue contrast to localise these interesting structures. Here are some example slices from a 3D MR volume:
![alt text](https://github.com/YipengHu/example-data/raw/master/label-reg-demo/media/volume_mr.jpg "MR Image Volume")
The catch is that the MR imaging is difficult and expensive to use _during_ those procedures, and requires more imaging time (you might need to wait half hour to get full multi-parametric sequences to reliably find where the tumour is), so they are usually available only _before_ the procedure. If the MR image could be aligned with the ultrasound image in real-time, all the information can then be "registered" from the MR to the ultrasound images, becoming useful during the ultrasound-guided procedures. It sounds like a good solution, but it turned out to be a very difficult problem, stimulating a decade-long research topic. The [journal articule][Hu2018b] provides many references and examples describing the detailed difficulties and attempted solutions. This tutorial describes an alternative using deep learning method. The main advantage is the resulting registration between the 3D data is fast (several 3D registrations in one second), fully-automated and easy to implement.

Using the code from this tutorial, one should be able to re-produce the entire method from training to inference and, possibly, (with a bit linking-up with a GUI of choice) deploying the learned model for some real-time surgical guidance! 


### <a name="section1-1"></a>Example Data
Due to medical data restrictions, this tutorial uses some fake (fewer and smaller) data to mimic those from the prostate imaging application. You can download these by clicking the following link:

[**Example data**][data]

First unzip the downloaded data to folders, which you need to specify in [config.py][config_file] in order to run the code.

In summary, for each numbered patient, there is a quartet of data, a 3D MR volume, a 3D ultrasound volume, several landmarks delineated from MR and ultrasound volumes. The landmark volumes are in 4D with the fourth dimention indicates different types of landmarks, such as the prostate gland and the apex/base points (where urethra enters and exists prostate gland).


## <a name="section2"></a>2 Weakly-Supervised Dense Correspondence Learning
The idea of the wearkly-supervised learning is to use expert labels that represent the same anatomical structures. Depending on one's personal viewpoint, this type of label-driven methods may be considered as being "lazy" (e.g. compared to simulating complex biological deformation or engineering sophisticated similarity measure, used in supervised or unsupervised approches, respectively) or being "industrious" as a great amount manually-annotated anatomical structures in volumetric data are requried.

While the goal is predicting DDF which we do not have ground-truth data for, the method is considered as "weakly-supervised" because the anatomical labels are used only in training. They are treated as if they are the "target labels" instead of "input predictors" in a classical regression analysis. Various formulations of the [weakly-supervised registration](#section4) is discussed and it is not in the papers! ;)

The trick here is to use _only_ images (moving and fixed) as input to the neural network without any labels, but the netowrk-predicted DDF can be used to transform the associated labels (from the same images) to match to each other, as shown in the picture:
![alt text](https://github.com/YipengHu/example-data/raw/master/label-reg-demo/media/training.jpg "Training")

As a result, labels are not requried in the subsequent inference, i.e. the registration predicting DDFs. It is, therfore, a fully-automated image registration accepts a pair of images and predicts a DDF, without segmentation of any kind to aid the alignment or even initialisation.
![alt text](https://github.com/YipengHu/example-data/raw/master/label-reg-demo/media/inference.jpg "Inference")


### <a name="section2-1"></a>Label Similarity Measures
The main problems with label-driven registration methods are labels representing corresponding structures are inherently sparse - among training cases, the same anatomical structures are not always present between a given moving and fixed image pair for training; when available, they neither cover the entire image domain nor detailed voxel correspondence. The following two examples show that 

![alt text](https://github.com/YipengHu/example-data/raw/master/label-reg-demo/media/landmarks_2cases.jpg "Example Landmarks")

Using cross-entropy to direct measure the loss between two given binary masks (representing the segmentation of the corresponding anatomies) has several problems:
1 - 

Here we adopted an efficient implementation of the differenciable Dice with spatial smoothing on labels to combat these problems. The entrance function for the implemented multi-scale Dice is *multi_scale_loss* in [losses.py][loss_file].

Unfortunately, the use of the Dice lost the intuitive interpretation of the statistical distribution assumed on the label matching. Further discussion is in [Section 4 Weakly-Supervised Registration Revisted](#section4). However, multi-scale Dice works well in practice.


### <a name="section2-2"></a>Convolutional Neural Networks for Predicting Displacements
The module [networks.py][network_file] implements the networks and some variants in the papers.


### <a name="section2-3"></a>Deformation Regularisation
The main function implementing bending energy is *compute_bending_energy* in [losses.py][loss_file].


### <a name="section2-4"></a>Training
* Training-Step-1 (data):
Get the data reader objects and some of useful information with in the readers:
```python
reader_moving_image, reader_fixed_image, reader_moving_label, reader_fixed_label = helper.get_data_readers(
    config.Data.dir_moving_image,
    config.Data.dir_fixed_image,
    config.Data.dir_moving_label,
    config.Data.dir_fixed_label)
```

* Training-Step-2 (graph):
Placeholders for both images and affine transformation parameters for data augmentation:
```python
ph_moving_image = tf.placeholder(tf.float32, [config.Train.minibatch_size]+reader_moving_image.data_shape+[1])
ph_fixed_image = tf.placeholder(tf.float32, [config.Train.minibatch_size]+reader_fixed_image.data_shape+[1])
ph_moving_affine = tf.placeholder(tf.float32, [config.Train.minibatch_size]+[1, 12])
ph_fixed_affine = tf.placeholder(tf.float32, [config.Train.minibatch_size]+[1, 12])
input_moving_image = util.warp_image_affine(ph_moving_image, ph_moving_affine)  # data augmentation
input_fixed_image = util.warp_image_affine(ph_fixed_image, ph_fixed_affine)  # data augmentation
```

Now, the registration network:
```python
reg_net = network.build_network(network_type=config.Network.network_type,
                                minibatch_size=config.Train.minibatch_size,
                                image_moving=input_moving_image,
                                image_fixed=input_fixed_image
                                )
```
The network predicting the DDF is only part of the graph, but separating it is useful in [infernce](#section2-5) stage.

The loss is constructed by first setting placeholders for label data (N.B. labels appear now and are not used in predicting DDF), with data augmentation using the same affine transformations:
```python
ph_moving_label = tf.placeholder(tf.float32, [config.Train.minibatch_size]+reader_moving_image.data_shape+[1])
ph_fixed_label = tf.placeholder(tf.float32, [config.Train.minibatch_size]+reader_fixed_image.data_shape+[1])
input_moving_label = util.warp_image_affine(ph_moving_label, ph_moving_affine)  # data augmentation
input_fixed_label = util.warp_image_affine(ph_fixed_label, ph_fixed_affine)  # data augmentation
```

Warp the moving labels using the predicted DDFs:
```python
warped_moving_label = reg_net.warp_image(input_moving_label)
```

Then the label similarity can be computed between the warped moving labels and the fixed labels:
```python
loss_similarity, loss_regulariser = loss.build_loss(similarity_type=config.Loss.similarity_type,
                                                    similarity_scales=config.Loss.similarity_scales,
                                                    regulariser_type=config.Loss.regulariser_type,
                                                    regulariser_weight=config.Loss.regulariser_weight,
                                                    label_moving=warped_moving_label,
                                                    label_fixed=input_fixed_label,
                                                    network_type=config.Network.network_type,
                                                    ddf=reg_net.ddf)
```


The Dice scores should be consistently above 0.9 after a few thousand iterations on the gland segmentation labels (for convinience, they are always the first landmark, i.e. label_index=0, in the example data. But this is not a requirement in the random stage sampling.)



The two files, [training.py][training_file] and [inference.py][inference_file], contain all the necessary codes to perform the training and inference described above with simple but sufficient file IO support. Read the next section for more information.


### <a name="section2-5"></a>Inference


## <a name="section3"></a>3 Try with Your Own Image-Label Data
Files readable by [NiBabel][nibabel] should work with the DataReader in [helpers.py][helper_file]. The quatets of matmul([[moving], [fixed]], [[image, label]]) data should be organised as follows in order to run the code without modification:
* The training data should be in saparate folders and the folder names are specified in the [config.py][config_file], for example:
```python
dir_moving_image = os.path.join(os.environ['HOME'], 'git/label-reg-demo/data/train/mr_images')
dir_fixed_image = os.path.join(os.environ['HOME'], 'git/label-reg-demo/data/train/us_images')
dir_moving_label = os.path.join(os.environ['HOME'], 'git/label-reg-demo/data/train/mr_labels')
dir_fixed_label = os.path.join(os.environ['HOME'], 'git/label-reg-demo/data/train/us_labels')
```
* They should have the same number of subjects, number of image volume files. The code currently assigns corresponding subjects by re-ordered file names. So it is easier to just rename them so that four files from the same patient/subject have the same file name;
* Each image file contains a 3D image of the same shape;
* Each label file contains a 4D volume with 4th dimension contains different landmarks delineated from the associated image volume. The segmented-foreground and background are represented by 0s and 1s, respectively;
* The number of landmarks can be variable (and large) across subjects/patients, but has to be the sanme between _moving label_ and _fixed label_ from the same subject/patient, i.e. corresponding landmark pairs;
* The image and each lanadmark (one 3D volume in the 4D label) should have the same shape, while the image and landmark do not need to have the same shape;
* If inference or test is needed, also specify those folder names in Inference [config.py][config_file].

That's it. Let me know if any problem.


## <a name="section4"></a>4 Weakly-Supervised Registration Revisted
First, the weakly-supervised registartion network training reflects an end of the human-label-driven vs unsupervised-pattern-recognition spectum. The prostate imaging application here is a typical exmaple that classical intensity-based similarity measures such as mutual information are simply do not work.


[data]: https://github.com/YipengHu/example-data/raw/master/label-reg-demo/data.zip

[config_file]: ./config.py
[loss_file]: ./labelreg/losses.py
[network_file]: ./labelreg/networks.py
[helper_file]: ./labelreg/helpers.py
[inference_file]: ./inference.py
[training_file]: ./training.py

[Hu2018a]: https://arxiv.org/abs/1711.01666
[Hu2018b]: https://arxiv.org/abs/1711.01666

[niftynet]: http://niftynet.io/
[nibabel]: http://nipy.org/nibabel/


