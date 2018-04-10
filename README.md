# Weakly-Supervised Convolutional Neural Networks for Multimodal Image Registration


## Introduction
This tutorial aims to provide minimum and self-explanatory scripts to re-work the implementation of a deep-learning-based image registration method in [Hu et al 2018][Hu2018a] (and the preliminary work in [Hu et al ISBI2018][Hu2018b]). An efficient re-implementation with other utilities is available at [NiftyNet Platform][niftynet]. The sections are organised as follows:

* [**1. Multimodal Image Registration**](#section1)
* [     - Example Data](#section1-1)
* [**2. Weakly-Supervised Dense Correspondence Learning**](#section2)
* [     - Label Similarity Measures](#section2-1)
* [     - Convolutional Neural Networks for Predicting Displacements](#section2-2)
* [     - Deformation Regularisation](#section2-3)
* [     - Training](#section2-4)
* [     - Inference](#section2-5)
* [**3. Try with Your Own Image-Label Data**](#section3)
* [**4. Weakly-Supervised Registration Revisited**](#section4)

Using the code from this tutorial, one should be able to re-produce the entire method from training to inference and, possibly (with a bit linking-up with a GUI of choice), to deploy the learned model for some real-time surgical guidance! 


## <a name="section1"></a>1 Multimodal Image Registration
Medical image registration aims to find a dense displacement field (DDF), such that a given "moving image" can be warped (transformed using the predicted DDF) to match a second "fixed image", that is, the corresponding anatomical structures are aligned in the same spatial locations. A DDF is typically a set of displacements (in x-, y- and z components) defined at every voxel in the fixed image coordinates, so each set is about three times of the size of the fixed image. They usually are inverted displacemnts (i.e. from fixed to moving) to resample intensities from fixed image (which is the one being _warped_ to moving image coordinates). 

The definition of "multimodal" varies from changing of imaging parameters (such as MR sequencing or contrast agents) to images from different scanners. An example application is to support 3D-ultrasound-guided intervention for prostate cancer patients. The 3D ultrasound image looks like this:
![alt text](https://github.com/YipengHu/example-data/raw/master/label-reg-demo/media/volume_us.jpg "Ultrasound Image Volume")
In these procedures, the urologists would like to know where the suspicious regions are, so they can take a tissue sample to confirm pathology, i.e. biopsy, or they may treat certain small areas such as a tumour, i.e. focal therapy. They also want to avoid other healthy delicate surrounding or internal structures such as nerves, urethra, rectum and bladder. However, from ultrasound imaging, it is hard enough to work out the boundaries of the prostate gland, let alone the detailed anatomical and pathological structures. 

MR imaging, on the other hand, provides a better tissue contrast to localise these interesting structures. Here are some example slices from a 3D MR volume:
![alt text](https://github.com/YipengHu/example-data/raw/master/label-reg-demo/media/volume_mr.jpg "MR Image Volume")
The catch is that the MR imaging is difficult and expensive to use _during_ those procedures. For examples, they are susceptible to metal instruments and require longer imaging time (you might need to wait half hour to get full multi-parametric sequences to reliably find where the tumour is), so they are usually only available _before_ the procedure. If the MR image can be spatially aligned with the ultrasound image in real-time, all the information can then be "registered" from the MR to the ultrasound images, very useful during the ultrasound-guided procedures. It sounds like a good solution, but it turned out to be a very difficult problem, stimulating a decade-long research topic. The [journal article][Hu2018a] provides many references and examples describing the detailed difficulties and attempted solutions. This tutorial describes an alternative using deep learning method. The main advantage is the resulting registration between the 3D data is fast (several 3D registrations in one second), fully-automated and easy-to-implement.

In this set-up, the ultrasound images will be considered as the _fixed image_, while the MR image will be the _moving image_. It is only a practical choice to avoid unecessary extrapolation, such that, when warping, the intensities from the image with larger field-of-view (usually the case of MR) can be re-sampled at the (ultrasound) image coordinates that cover smaller field-of-view.


### <a name="section1-1"></a>Example Data
Due to medical data restrictions, this tutorial uses some fake (fewer and smaller) data to mimic those from the prostate imaging application. You can download these by clicking the following link:

[**Download Example Data**][data]

First unzip the downloaded data to folders, which you need to specify in [config_demo.ini][config_file] in order to run the application.

In summary, for each numbered patient, there is a quartet of data, a 3D MR volume, a 3D ultrasound volume, several landmarks delineated from MR and ultrasound volumes. The landmark volumes are in 4D with the fourth dimension indicates different types of landmarks, such as the prostate gland and the apex/base points (where urethra enters and exists prostate gland).


## <a name="section2"></a>2 Weakly-Supervised Dense Correspondence Learning
The idea of the weakly-supervised learning is to use expert labels that represent the same anatomical structures. Depending on one's personal viewpoint, this type of label-driven methods can be considered as being "lazy" (e.g. compared to simulating complex biological deformation or engineering sophisticated similarity measure, used in supervised or unsupervised approaches, respectively) or being "industrious" as a great amount manually-annotated anatomical structures in volumetric data are required.

While the goal is predicting DDF which we do not have ground-truth data for, the method is considered as "weakly-supervised" because the anatomical labels are used only in training. They are treated as if they are the "target labels" instead of "input predictors" in a classical regression analysis. More on the topic of [weakly-supervised registration](#section4) is discussed and it is not in the papers! ;)

The trick here is to use _only_ images (moving and fixed) as input to the neural network without any labels, but the network-predicted DDF can be used to transform the associated labels (from the same images) to match to each other, as shown in the picture:
![alt text](https://github.com/YipengHu/example-data/raw/master/label-reg-demo/media/training.jpg "Training")

As a result, labels are not required in the subsequent inference, i.e. the registration predicting DDFs. It is, therefore, a fully-automated image registration accepts a pair of images and predicts a DDF, without segmentation of any kind to aid the alignment or even initialisation.
![alt text](https://github.com/YipengHu/example-data/raw/master/label-reg-demo/media/inference.jpg "Inference")


### <a name="section2-1"></a>Label Similarity Measures
If we had landmarks as small as image voxels densely populated across the image domain, learning dense correspondence (also represented as DDF) becomes a supervised learning problem. The main problem with anatomical-label-driven registration methods is that anatomical labels representing corresponding structures are inherently sparse. Among training cases, the same anatomical structures are not always present on a moving image and on a fixed image; when available, they neither cover the entire image domain nor detailed voxel correspondence. The following two examples show that, for these two cases, prostate glands in blue (possibly the most consistent landmarks that are only ones always available across all cases) may be found on both images for both cases, while _patient-specific_ cysts (orange in the left case) or calcifications (yellow in the right case) can only be identified on a case-by-case basis:

![alt text](https://github.com/YipengHu/example-data/raw/master/label-reg-demo/media/landmarks_2cases.jpg "Example Landmarks")

An efficient implementation of the differentiable Dice with spatial smoothing on labels is used here to deal with the label sparsity. The entrance function for the implemented multi-scale Dice is *multi_scale_loss* in [losses.py][loss_file].

Although the use of the Dice lost the intuitive interpretation of the statistical distribution assumed on the _weak labels of correspondence_, multi-scale Dice works well in practice as a generic loss function. Further discussion is in [Section 4 Weakly-Supervised Registration Revisited](#section4).


### <a name="section2-2"></a>Convolutional Neural Networks for Predicting Displacements
The module [networks.py][network_file] implements the networks and some variants in the papers. The network architecture is a modified encoder-decoder convolutional neural network with two features:
* Several types of shortcut layers, resnet, summation skip layers, additive trilinear upsampling;
* Additive output layers to output a single DDF with shortcuts directly to each resolution level of the entire network.
The additive output layers can be configured by overwriting the default argument _ddf_levels_ of _LacalNet_ class in [networks.py][network_file].

![alt text](https://github.com/YipengHu/example-data/raw/master/label-reg-demo/media/network_architecture.jpg "Network Architecture")


### <a name="section2-3"></a>Deformation Regularisation
Partly due to the sparsity of the training labels, regularisation of the predicted deformation files is essential. Borrowing the regularisation strategy used in traditional image registration algorithms as well as those from classical mechanics, usually, smoothness of the entire displacement field is penalised in addition to the [label similarity measures](#section2-1). In essence, these functions measure how non-smooth the DDF is, based on the first- and/or second order derivatives of the displacemnt w.r.t. the spatial (image) coordinates. The main function implementing bending energy is *compute_bending_energy* in [losses.py][loss_file], among other choices.


### <a name="section2-4"></a>Training
* **Training-Step-1 (data)**:
Get the data reader objects and some of useful information with the readers:

```python
reader_moving_image, reader_fixed_image, reader_moving_label, reader_fixed_label = helper.get_data_readers(
    '~/git/label-reg-demo/data/train/mr_images',
    '~/git/label-reg-demo/data/train/us_images',
    '~/git/label-reg-demo/data/train/mr_labels',
    '~/git/label-reg-demo/data/train/us_labels')
```

* **Training-Step-2 (graph)**:
Placeholders for **4** pairs of moving and fixed images and affine transformation parameters with size of **[1, 12]** for data augmentation:

```python
ph_moving_image = tf.placeholder(tf.float32, [4]+reader_moving_image.data_shape+[1])
ph_fixed_image = tf.placeholder(tf.float32, [4]+reader_fixed_image.data_shape+[1])
ph_moving_affine = tf.placeholder(tf.float32, [4]+[1, 12])
ph_fixed_affine = tf.placeholder(tf.float32, [4]+[1, 12])
input_moving_image = util.warp_image_affine(ph_moving_image, ph_moving_affine)  # data augmentation
input_fixed_image = util.warp_image_affine(ph_fixed_image, ph_fixed_affine)  # data augmentation
```

The **local** registration network described in [Convolutional Neural Networks for Predicting Displacements](#section2-2) is the default architecture:

```python
reg_net = network.build_network('local',
                                minibatch_size=4,
                                image_moving=input_moving_image,
                                image_fixed=input_fixed_image)
```
The network predicting the DDF is only part of the so-called _computation graph_, but separating it is useful in [inference](#section2-5) stage.

The loss function is constructed by first setting placeholders for label data (N.B. labels appear now and are not used in predicting DDF), with data augmentation using the same affine transformations:

```python
ph_moving_label = tf.placeholder(tf.float32, [4]+reader_moving_image.data_shape+[1])
ph_fixed_label = tf.placeholder(tf.float32, [4]+reader_fixed_image.data_shape+[1])
input_moving_label = util.warp_image_affine(ph_moving_label, ph_moving_affine)  # data augmentation
input_fixed_label = util.warp_image_affine(ph_fixed_label, ph_fixed_affine)  # data augmentation
```

Warp the moving labels using the predicted DDFs:

```python
warped_moving_label = reg_net.warp_image(input_moving_label)
```

The [label similarity measure](#section2-1) and the weighted [deformation regularisation](#section2-3) can then be constructed between the warped moving labels and the fixed labels:

```python
loss_similarity, loss_regulariser = loss.build_loss(similarity_type='dice',
                                                    similarity_scales=[0,1,2,4,8,16],
                                                    regulariser_type='bending',
                                                    regulariser_weight=0.5,
                                                    label_moving=warped_moving_label,
                                                    label_fixed=input_fixed_label,
                                                    network_type='local',
                                                    ddf=reg_net.ddf)
```

* **Training-Step-3** (Optimisation):
Adam is the favourite optimiser here with default settings with an initial learning rate around 1e-5 (it may need to be tuned down if the network_type is set to 'global' or 'composite'):

```python
train_op = tf.train.AdamOptimizer(1e-5).minimize(loss_similarity+loss_regulariser)
```

The main iterative minibatch gradient descent is fairly standard, except for the two-stage clustering sampling when sampling the label data, after the image pairs being sampled. This is useful because different image pairs have different numbers of labels, a consequence for exploiting _ad hoc_ anatomical labels, which was further discussed in the papers. Importantly, this can form a compact training minibatch of image-label quartets with a fixed size, as defined in the placehoders in the previous steps, enabling efficient use of the parallel computing resource. 

A standard minibatch optimisation can be trivially modified to uniformly sample a label pair (moving and fixed) from those available label pairs delineated on each (first-stage) sampled image pair (i.e. moving and fixed images from one patient/subject). Assuming a minibatch size of 4 again, _num_labels[i]_ is the number of available labels for ith image pair:

```python
    # in each iteration step
    minibatch_idx = step % num_minibatch
    case_indices = train_indices[minibatch_idx*4:(minibatch_idx+1)*4]
    label_indices = [random.randrange(reader_moving_label.num_labels[i]) for i in case_indices]
```

where:
```python
num_minibatch = int(reader_moving_label.num_data/4)
train_indices = [i for i in range(reader_moving_label.num_data)]
```


Two utility computing nodes are also included for monitoring the training process. Here, the binary Dice, _dice_, and distance between centroids, _dist_, are implemented in [utils.py][util_file]. 

The Dice scores should be consistently above 0.90 after a few thousand iterations on the gland segmentation labels (for convenience, gland segmentations are always the first landmark, i.e. label_index=0, in the example data. But this is not a requirement.) The top-level file, [training.py][training_file] contains all the necessary code to perform the training described above with simple file IO support. Read the [Section 3](#section3) for more information on how to run the code with real imaging data.


### <a name="section2-5"></a>Inference
Considering the difference between the inference and the training is an effective way to obtain insight of the registration method. The first difference is on the data. While the training requires moving-fixed-image-label quartets, inference only needs a pair of moving and fixed images:

```python
reader_moving_image, reader_fixed_image, _, _ = helper.get_data_readers('~/git/label-reg-demo/data/test/mr_images',
                                                                        '~/git/label-reg-demo/data/test/us_images')
ph_moving_image = tf.placeholder(tf.float32, [reader_moving_image.num_data]+reader_moving_image.data_shape+[1])
ph_fixed_image = tf.placeholder(tf.float32, [reader_fixed_image.num_data]+reader_fixed_image.data_shape+[1])
```

First, restore the trained network (only the _reg_net_ for predicting DDF and trained weights):

```python
reg_net = network.build_network(network_type='local',
                                minibatch_size=reader_moving_image.num_data,
                                image_moving=ph_moving_image,
                                image_fixed=ph_fixed_image)

saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, '~/git/label-reg-demo/data/model.ckpt')
```

Then, the DDF can be computed using one-pass evaluation on the pairs of images to register:

```python
testFeed = {ph_moving_image: reader_moving_image.get_data(),
            ph_fixed_image: reader_fixed_image.get_data()}
ddf = sess.run(reg_net.ddf, feed_dict=testFeed)
```
And, that's it for inference.


Depending on the application, the predicted DDF can be used in several ways, such as a) warping the moving MR images given a real-time ultrasound image, 2) warping a binary image representing segmentation(s) on the moving MR image (such as a tumour), 3) transforming the real-time information (such as surgical instrument position) back to the MR image space or 4) transforming a MR-defined point cloud to ultrasound imaging coordinates (N.B. in this case an inverting of the predicted transformation may be required). This tutorial demonstrates a function using TensorFlow ([apps.py][app_file]) to warp MR images or MR-defined labels in batches on GPU:

```python
warped_images = app.warp_volumes_by_ddf(reader_moving_image.get_data(), ddf)

data_moving_label = helper.DataReader('~/git/label-reg-demo/data/test/mr_labels').get_data(label_indices=[0])
warped_labels = app.warp_volumes_by_ddf(data_moving_label, ddf)
```

Example code is in the top-level [inference.py][inference_file], which can also be used as an application for [your own image-label data](#section3).


## <a name="section3"></a>3 Try with Your Own Image-Label Data
[TensorFlow][tensorflow_install] needs to be installed first, with a handful standard python modules, numpy, random, os, sys, time and nibabel (for file IO only), all easily available if not already installed. Get a copy of the code, e.g. on linux:

```
git clone https://gitlab.com/yipeng/label-reg-demo
```

or:

[**Download Code** (zip file)][code]


Data files readable by [NiBabel][nibabel] should work with the DataReader in [helpers.py][helper_file]. The quartets of moving-and-fixed image-and-label data should be organised as follows in order to run the code without modification:

* The training data should be in separate folders and the folder names are specified under the [Data] section in the [config_demo.ini][config_file], for example:

```
[Data]
dir_moving_image = ~/git/label-reg-demo/data/train/mr_images
dir_fixed_image = ~/git/label-reg-demo/data/train/us_images
dir_moving_label = ~/git/label-reg-demo/data/train/mr_labels
dir_fixed_label = ~/git/label-reg-demo/data/train/us_labels
```

* They should have the same number of image volume files (patients/subjects). The code currently assigns corresponding subjects by re-ordered file names. So it is easier to just rename them so that four files from the same patient/subject have the same file name;
* Each image file contains a 3D image of the same shape, with a data type convertable to float32;
* Each label file contains a 4D volume with 4th dimension contains different landmarks delineated from the associated image volume. The segmented-foreground and background are represented by or convertable to float32 0s and 1s, respectively;
* The number of landmarks can be variable (and large) across patients/subjects, but has to be the same within each pair from the same patient/subject, between _moving label_ and _fixed label_ (i.e. representing corresponding landmark pairs);
* The image and each of its landmark (one 3D volume in the 4D label) should have the same shape, while the moving and fixed data do not need to have the same shape;
* If inference or test is needed, also specify those folder names in Inference [config_demo.ini][config_file].

One can customise a config file to specify other parameters mentioned in the tutorial. Use the same [config_demo.ini][config_file] file as a template. Both [training.py][training_file] and [inference.py][inference_file] can take a command line argument for the customised config file path, for example:

```python
python3 ~/git/label-reg-demo/training.py ~/myTrainingConfig.ini
```

The trained model will be saved in file_model_save under [Train] section specified in [config_demo.ini][config_file].

```python
python3 ~/git/label-reg-demo/inference.py ~/myInferenceConfig.ini
```

For demo purpose, three files, the ddf, warped_image and warped_label (if dir_moving_label is supplied) will be saved in dir_save under [Inference] section specified in [config_demo.ini][config_file].


That's it. Let me know if any problem.


## <a name="section4"></a>4 Weakly-Supervised Registration Revisited
First, **weakly-supervised learning** is not a rigorously defined term. It was not mentioned at all in [Ian Goodfellow's Deep Learning book][TheDeepLearningBook]. Strictly speaking, the registration method used here is still an unsupervised regression without the labels for targets (i.e. the displacements) that need to be predicted. The target labels of registration should be ground-truth displacement fields which are not easily available (the main point that motivates this method). An alternative form of displacement field is a "correspondence table" indicating where in the fixed image coordinates every voxel in moving image should move to. One way to go about the weak labels is to consider the anatomical labels arranged in such a table, but corrupted with non-i.i.d. noise. All the voxels in an anatomical region (defined by the segmentation labels) on one image, should be more likely to be in the same region on another image. With added nuisance of inconsistently available types of anatomical landmarks, a naive multi-class implementation of such a correspondence table would be very sparse. The initial work [Hu et al ISBI2018][Hu2018b] explored the idea to consider, instead, a two-class classification problem, where the abstract concept of correspondence is used, two voxels (after warping) are considered being correspondent or not correspondent. In this case, the segmentations of different types of anatomical regions become noise-corrupted _data_ of these correspondence, that is, being correspondent if both are foreground 1s or both are background 0s, not correspondent otherwise. A well-defined cross-entropy was used to measure the overall classification loss. 

The main problem with the two-class formulation is about weighting. The cross-entropy assumes no difference between voxel locations are nearer to boundaries and those are not. It does not distinguish difference between foreground and background, which can be substantially altered by imaging parameters (such as acquired fields of view), or which type of anatomical regions the (non)correspondence voxels come from. For example, a voxel in the foreground segmentation of the gland should, without any other information, be a _weaker_ correspondence label than that from a foreground voxel from very small landmark, as the latter is a very _strong_ predictor for where this voxel should move to, although it helps very little to indicate the displacement field everywhere else. This is why some heavy heuristic preprocessing label smoothing was used.

Overlap measures, such as Dice, have an interesting quality to re-weight between these classes, with the changing denominator of Dice (as overlap changes) acting as a "dynamic" weighting mechanism. Therefore, it has been adopted in medical image segmentation tasks with consistently superior performance (after all, it is the measure for evaluating segmentation). A similar effect has also been observed in the label-driven registration tasks. The multi-scale strategy is to mitigate the practical issues on the landmarks with smaller volumes, so they will still produce non-zero gradients even without overlap during training, as shown in the following picture with different types of landmarks being filtered at different scales:

![alt text](https://github.com/YipengHu/example-data/raw/master/label-reg-demo/media/multiscale.jpg "Multiscale Label Filtering") 

The downside of Dice, however, is that it lacks a clear interpretation of the weakly-supervision, leaning towards a general unsupervised learning where any loss function is sensible if it drives the image alignment. The practical difference worth a name of "weak supervision" is perhaps that the loss function is not dependent on the image modality, but applied on segmentation labels which, to certain degree, is closer to traditional feature-based registration method. The neural network is a better way to learn the feature representation. It also reflects the fact that this method, compared with other unsupervised learning, relies on anatomical knowledge in human labelling instead of statistical properties summarised otherwise (e.g. through image-based similarity measures).

It may be the case that, due to the nature of weakly-supervised learning, different formulations of the loss function, or the combination of the loss and the regularisation, is only a different weighting strategy to "guess" the dense correspondence. Without ground-truth (for training and validation), this will inherently be dependent on image modalities and applications. Therefore, it may be better to investigate application-specific loss function (such as better surrogates of the true TREs on regions of interest).

Even with unlimited data pairs, there is a physical bounds of the label availability partly due to the underlying imaging process that simply do not produce voxel-level correspondence information and partly due to limited anatomical knowledge. In this case, prior knowlege on application-specific physical transformation (instead of bending energy for instance), other intensity-based similarity and predicting with labels might provide further assitance.


[data]: https://github.com/YipengHu/example-data/raw/master/label-reg-demo/data.zip
[code]: https://gitlab.com/Yipeng/label-reg-demo/repository/master/archive.zip

[config_file]: ./config_demo.ini
[loss_file]: ./labelreg/losses.py
[network_file]: ./labelreg/networks.py
[helper_file]: ./labelreg/helpers.py
[util_file]: ./labelreg/utils.py
[inference_file]: ./inference.py
[training_file]: ./training.py
[app_file]: ./labelreg/apps.py

[Hu2018a]: https://arxiv.org/abs/1711.01666
[Hu2018b]: https://arxiv.org/abs/1711.01666
[TheDeepLearningBook]: http://www.deeplearningbook.org/

[niftynet]: http://niftynet.io/
[nibabel]: http://nipy.org/nibabel/
[tensorflow_install]: https://www.tensorflow.org/install/


