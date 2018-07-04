# Weakly-Supervised Convolutional Neural Networks for Multimodal Image Registration


## Introduction
This demo provides minimum and self-explanatory scripts to re-work the implementation of a deep-learning-based image registration method in [Hu et al 2018][Hu2018a] (and the preliminary work in [Hu et al ISBI2018][Hu2018b]). A re-implementation with other utilities is also available at [NiftyNet Platform][niftynet]. The sections are organised as follows:

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

Using this code, one should be able to re-produce the entire method from training to inference and, possibly (with a bit linking-up with a GUI of choice), to deploy the learned model for some real-time surgical guidance! 


## <a name="section1"></a>1 Multimodal Image Registration
Medical image registration aims to find a dense displacement field (DDF), such that a given "moving image" can be warped (transformed or spatially resampled using the predicted DDF) to match a second "fixed image". _Matching_ in the example application means the same anatomical structures are aligned at the same spatial locations, technically also known as establishing dense correspondence. A DDF is typically a set of displacements (in x-, y- and z components) defined at every voxel in the fixed image coordinates, so each set is about three times of the size of the fixed image. They usually are inverted displacements (i.e. from fixed to moving) to resample intensities from fixed image (which is the one being _warped_ to moving image coordinates). 

The definition of "multimodal" varies from changing of imaging parameters (such as MR sequencing or contrast agents) to images from different scanners. An example application is to support 3D-ultrasound-guided intervention for prostate cancer patients. The 3D ultrasound images look like this:
![alt text](https://github.com/YipengHu/example-data/raw/master/label-reg-demo/media/volume_us.jpg "Ultrasound Image Volume")
In these procedures, the urologists would like to know where the suspicious regions are, so they can take a tissue sample to confirm pathology, i.e. biopsy, or they may treat certain small areas such as a tumour, i.e. focal therapy. They also want to avoid other healthy delicate surrounding or internal structures such as nerves, urethra, rectum and bladder. However, from ultrasound imaging, it is hard enough to work out the boundaries of the prostate gland, let alone the detailed anatomical and pathological structures. 

MR imaging, on the other hand, provides a better tissue contrast (for human eyes anyway) to localise these interesting structures. Here are some example slices from a 3D MR volume:
![alt text](https://github.com/YipengHu/example-data/raw/master/label-reg-demo/media/volume_mr.jpg "MR Image Volume")
The catch is that the MR imaging is difficult and expensive to use _during_ those procedures. For examples, they are susceptible to metal instruments and require longer imaging time (you might need to wait half hour to get full multi-parametric sequences to reliably find where the tumour is, that is even before the radiologist's full report), so they are usually only available _before_ the procedure. If the MR image can be spatially aligned with the ultrasound image in real-time, all the information can then be "registered" from the MR to the ultrasound images, very useful during the ultrasound-guided procedures. It sounds like a good solution, but turned out to be a difficult problem, stimulating a decade-long research topic. The [technical article][Hu2018a] provides many references and examples describing the detailed difficulties and attempted solutions. This demo describes an alternative using deep learning. The main advantage is that the resulting registration between the 3D data is fast (several 3D registrations in one second), fully-automated and easy-to-implement.

In this application, the ultrasound images will be considered as the _fixed image_, while the MR image will be the _moving image_. It is only a practical choice to avoid unnecessary extrapolation, such that, when warping, the intensities from the image with larger field-of-view (usually the case of MR) can be resampled at the (ultrasound) image coordinates that cover smaller field-of-view.


### <a name="section1-1"></a>Example Data
This demo uses some make-believe (fewer, smaller and blurred) data for tutorial purpose. They can be downloaded by clicking the following link:

[**Download Example Data**][data]

First unzip the downloaded data to folders, which you need to specify in [config_demo.ini][config_file] in order to run the application.

In summary, for each numbered patient, there is a quartet of data, a 3D MR volume, a 3D ultrasound volume, several anatomical landmark pairs delineated from MR and ultrasound volumes. The landmark volumes are in 4D with the fourth dimension indicates different types of landmarks, such as the prostate gland and the apex/base points (where urethra enters and exists prostate gland).


## <a name="section2"></a>2 Weakly-Supervised Dense Correspondence Learning
The idea of the weakly-supervised learning is to use labels that represent the same anatomical structures. Depending on one's viewpoint, this type of label-driven methods can be considered as being "lazy" (e.g. compared to simulating complex biological deformation or engineering sophisticated similarity measure, used in supervised or unsupervised approaches, respectively) or as being an "industrious effort" as an non-trivial amount of manually-annotated anatomical structures in volumetric data are required.

While the goal of registration is predicting DDF which we do not have ground-truth data for, the method is considered as "weakly-supervised" because the anatomical labels are used only in training. They are treated as if they are the "target labels" instead of "input predictors" in a classical regression analysis. More on the topic of [weakly-supervised registration](#section4) is discussed and it is not in the papers! ;)

The trick here is to use _only_ images (moving and fixed) as input to the neural network without any labels, but the network-predicted DDF can be used to warp the associated labels (from the moving image) to match to labels from the target image, as shown in the picture:
![alt text](https://github.com/YipengHu/example-data/raw/master/label-reg-demo/media/training.jpg "Training")

As a result, labels are not required in the subsequent inference, i.e. the registration predicting DDFs. It is, therefore, a fully-automated image registration accepts a pair of images and predicts a DDF, without segmentation of any kind to aid alignment or initialisation.
![alt text](https://github.com/YipengHu/example-data/raw/master/label-reg-demo/media/inference.jpg "Inference")


### <a name="section2-1"></a>Label Similarity Measures
If we had landmarks as small as image voxels densely populated across the image domain, learning dense correspondence (also represented as DDF) becomes a supervised learning problem. The challenge is that anatomical labels representing corresponding structures are inherently sparse. Among training cases, the same anatomical structures are not always present on a moving image and on a fixed image; when available, they neither cover the entire image domain nor detailed voxel correspondence. The following show example types of landmarks that can be found and reliably delineated in real cases, prostate glands in blue, a cyst in orange (left) and a cluster of calcification in yellow (right). While the prostate glands (possibly the most consistent and the only landmarks that are always available across all cases) may be found on both images for both cases, _patient-specific_ cysts or calcifications can only be identified on a case-by-case basis.
![alt text](https://github.com/YipengHu/example-data/raw/master/label-reg-demo/media/landmarks_2cases.jpg "Example Landmarks")

An efficient implementation of the differentiable Dice with spatial smoothing on labels is used here to deal with the label sparsity. The entrance function for the implemented multi-scale Dice is *multi_scale_loss* in [losses.py][loss_file].

Although the use of the Dice lost the intuitive interpretation of the statistical distribution assumed on the _weak labels of correspondence_, multi-scale Dice works well in practice as a generic loss function. More on this is discussed in [Section 4 Weakly-Supervised Registration Revisited](#section4).


### <a name="section2-2"></a>Convolutional Neural Networks for Predicting Displacements
_Registration Network_ used here has a subtly different meaning, compare with those considering all layers having gradient passing through as a part of the network. Conceptually, a registration network is only the part of computation graph (in TensorFlow language) that accepts a pair of images and predicts a DDF. The entire graph has other components/layers to construct the loss function, including warping labels and computing multiscale Dice for instance. 

The module [networks.py][network_file] implements the main registration network and some variants. The network architecture is a modified encoder-decoder convolutional neural network with two features:
* Several types of shortcut layers, resnet, summation skip layers, additive trilinear upsampling;
* Additive output layers to output a single DDF with shortcuts directly to each resolution level of the entire network.

The details of some motivation of the architectural features are explained in the paper. More options, such as the additive output layers, can be configured by overwriting the default argument _ddf_levels_ of _LocalNet_ class in [networks.py][network_file], which is the default _network_type_ capable of predicting nonrigid displacements.
![alt text](https://github.com/YipengHu/example-data/raw/master/label-reg-demo/media/network_architecture.jpg "Network Architecture")


### <a name="section2-3"></a>Deformation Regularisation
Partly due to the sparsity of the training labels, regularisation of the predicted deformation field is essential. Borrowing the regularisation strategy used in traditional image registration algorithms as well as those from classical mechanics, smoothness of the entire displacement field is penalised in addition to the [label similarity measures](#section2-1). In essence, these functions measure how non-smooth the DDF is, based on the first- and/or second order derivatives of the displacement w.r.t. the spatial (image) coordinates. The main function implementing default _regulariser_type_ **bending energy** can be found in *compute_bending_energy* in [losses.py][loss_file], among other choices of regularisation.


### <a name="section2-4"></a>Training
* **Training-Step-1 (Data)**:
First, get the data reader objects and other data information with the readers:
```python
reader_moving_image, reader_fixed_image, reader_moving_label, reader_fixed_label = helper.get_data_readers(
    '~/git/label-reg/data/train/mr_images',
    '~/git/label-reg/data/train/us_images',
    '~/git/label-reg/data/train/mr_labels',
    '~/git/label-reg/data/train/us_labels')
```

* **Training-Step-2 (Graph)**:
Placeholders for **4** pairs of moving and fixed images and affine transformation parameters with size of **[1, 12]** for data augmentation:
```python
ph_moving_image = tf.placeholder(tf.float32, [4]+reader_moving_image.data_shape+[1])
ph_fixed_image = tf.placeholder(tf.float32, [4]+reader_fixed_image.data_shape+[1])
ph_moving_affine = tf.placeholder(tf.float32, [4]+[1, 12])
ph_fixed_affine = tf.placeholder(tf.float32, [4]+[1, 12])
input_moving_image = util.warp_image_affine(ph_moving_image, ph_moving_affine)  # data augmentation
input_fixed_image = util.warp_image_affine(ph_fixed_image, ph_fixed_affine)  # data augmentation
```

The **local** registration network [(Convolutional Neural Networks for Predicting Displacements)](#section2-2) is the default architecture:
```python
reg_net = network.build_network('local',
                                minibatch_size=4,
                                image_moving=input_moving_image,
                                image_fixed=input_fixed_image)
```
The benefit of separating the registration network from the other parts of _computation graph_ will be seen in the later [inference](#section2-5) stage.

The loss function is constructed by first setting placeholders for label data (N.B. labels appear now and are not used in predicting DDF), with data augmentation using the same affine transformations:
```python
ph_moving_label = tf.placeholder(tf.float32, [4]+reader_moving_image.data_shape+[1])
ph_fixed_label = tf.placeholder(tf.float32, [4]+reader_fixed_image.data_shape+[1])
input_moving_label = util.warp_image_affine(ph_moving_label, ph_moving_affine)  # data augmentation
input_fixed_label = util.warp_image_affine(ph_fixed_label, ph_fixed_affine)  # data augmentation
```

Then, warp the moving labels using the predicted DDFs:
```python
warped_moving_label = reg_net.warp_image(input_moving_label)
```

Finally, the [label similarity measure](#section2-1) and the weighted [deformation regularisation](#section2-3) can then be constructed between the warped moving labels and the fixed labels:

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
The default _similarity_scales_ (in unit of voxel) are fewer than what used for real data as the reduced size of the example data. For a rule of thumb, this may be chosen so the 3 times of the largest standard deviation can cover the majority of the image domain. The _regulariser_weight_ will be dependent on data and application, such as how sparse the training labels are.

* **Training-Step-3 (Optimisation)**:
Adam is the favourite optimiser here with default settings with an initial learning rate around 1e-5, which may need to be tuned down if the network_type is set to 'global' or 'composite'.

```python
train_op = tf.train.AdamOptimizer(1e-5).minimize(loss_similarity+loss_regulariser)
```

The main iterative minibatch gradient descent is fairly standard, except for the two-stage clustering sampling when sampling the label data, after the image pairs being sampled. This is useful because different image pairs have different numbers of labels, a consequence for exploiting _ad hoc_ anatomical labels (further discussed in the papers). Importantly, this can form a compact training minibatch of image-label quartets with a fixed size, as defined in the placeholders in the previous steps, enabling efficient use of the parallel computing resource. 

A standard minibatch optimisation can be trivially modified to uniformly sample a _label pair_ (moving and fixed) from those available label pairs delineated on each (first-stage-sampled) _image pair_ (i.e. moving and fixed images from one patient/subject). Assuming a minibatch size of 4, _num_labels[i]_ is the number of available labels for ith image pair:
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

The Dice scores should be consistently above 0.90 after a few thousand iterations on the gland segmentation labels (for convenience, gland segmentations are always the first landmark, i.e. _label_index=0_, in the example data. But this is not a requirement and will be shuffled before feeding.) The top-level scripts, [training.py][training_file] contains all the necessary training code with simple file IO support. Read the [Section 3](#section3) for more information on how to run the code with real imaging data.


### <a name="section2-5"></a>Inference
Considering the difference between the inference and the training is an effective way to obtain insight of this registration method. The first difference is on the data. While the training requires moving-fixed-image-label quartets, inference only needs a pair of moving and fixed images:
```python
reader_moving_image, reader_fixed_image, _, _ = helper.get_data_readers('~/git/label-reg/data/test/mr_images',
                                                                        '~/git/label-reg/data/test/us_images')
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
saver.restore(sess, '~/git/label-reg/data/model.ckpt')
```

Then, the DDF can be computed using one-pass evaluation on the pairs of images to register:
```python
testFeed = {ph_moving_image: reader_moving_image.get_data(),
            ph_fixed_image: reader_fixed_image.get_data()}
ddf = sess.run(reg_net.ddf, feed_dict=testFeed)
```
Technically, that's it for inference.


Depending on the application, the predicted DDF can be used in several ways, such as **a)** warping the moving MR images given a real-time ultrasound image, **b)** warping a binary image representing segmentation(s) on the moving MR image (such as a tumour), **c)** transforming the real-time information (such as surgical instrument position) back to the MR image space or **d)** transforming a MR-defined point cloud to ultrasound imaging coordinates (N.B. in this case an inverting of the predicted transformation may be required). This tutorial also demonstrates a function _warp_volumes_by_ddf_ using TensorFlow ([apps.py][app_file]) to warp MR images or MR-defined labels in batches on GPU:
```python
warped_images = app.warp_volumes_by_ddf(reader_moving_image.get_data(), ddf)

data_moving_label = helper.DataReader('~/git/label-reg/data/test/mr_labels').get_data(label_indices=[0])
warped_labels = app.warp_volumes_by_ddf(data_moving_label, ddf)
```

Example inference script is in the top-level [inference.py][inference_file], which can also be used as an application for [your own image-label data](#section3).


## <a name="section3"></a>3 Try with Your Own Image-Label Data
**Obtain a copy of the code**

[TensorFlow][tensorflow_install] needs to be installed first, with a handful standard python modules, numpy, random, os, sys, time and nibabel (for file IO only), all easily available if not already installed. Get a copy of the code, e.g. on Linux:
```
git clone https://github.com/yipenghu/label-reg
```
or download from here:

[**Download Code** (zip file)][code]


**Prepare images and labels**

Data files readable by [NiBabel][nibabel] should work with the DataReader in [helpers.py][helper_file]. The quartets of moving-and-fixed image-and-label data should be organised as follows in order to run the code without modification:

* The training data should be in separate folders and the folder names are specified under the [Data] section in the [config_demo.ini][config_file], for example:
```
[Data]
dir_moving_image = ~/git/label-reg/data/train/mr_images
dir_fixed_image = ~/git/label-reg/data/train/us_images
dir_moving_label = ~/git/label-reg/data/train/mr_labels
dir_fixed_label = ~/git/label-reg/data/train/us_labels
```
* They should have the same number of image volume files (patients/subjects). The code currently assigns corresponding subjects by re-ordered file names. So it is easier to just rename them so that four files from the same patient/subject have the same file name;
* Each image file contains a 3D image of the same shape, with a data type convertible to float32;
* Each label file contains a 4D volume with 4th dimension contains different landmarks delineated from the associated image volume. The segmented-foreground and background are represented by or convertible to float32 0s and 1s, respectively;
* The number of landmarks can be variable (and large) across patients/subjects, but has to be the same within each pair from the same patient/subject, between _moving label_ and _fixed label_ (i.e. representing corresponding landmark pairs);
* The image and each of its landmark (one 3D volume in the 4D label) should have the same shape, while the moving and fixed data do not need to have the same shape;
* If inference or test is needed, also specify those folder names under [Inference] section in [config_demo.ini][config_file];
* Any global coordinate systems defined in file headers are ignored.

One can customise a config file to specify other parameters mentioned in the tutorial. Use the same [config_demo.ini][config_file] file as a template. Both [training.py][training_file] and [inference.py][inference_file] can take a command line argument for the customised config file path, for example:
```python
python3 ~/git/label-reg/training.py ~/myTrainingConfig.ini
```
The trained model will be saved in file_model_save under [Train] section specified in [config_demo.ini][config_file].

```python
python3 inference.py  myInferenceConfig.ini
```
For demo purpose, three files, the ddf, warped_image and warped_label (if dir_moving_label under [Inference] is supplied in [config_demo.ini][config_file]) will be saved in dir_save under [Inference] section specified in [config_demo.ini][config_file].

For a list of currently available top-level options:
```python
python3 training.py -help
python3 inference.py -help
```


That's it. 

* A little disclaimer - this is a re-worked code based on the original experimental code, for readability, simplicity and some modularity, but with very little test. So, let me know if any problems.


## <a name="section4"></a>4 Weakly-Supervised Registration Revisited
First, **weakly-supervised learning** is not a rigorously defined term. It was not mentioned at all in [Ian Goodfellow's Deep Learning book][TheDeepLearningBook]. Strictly speaking, the registration method used here is still an unsupervised regression without the labels for targets (i.e. the displacements) that need to be predicted. The target labels of registration should be ground-truth displacement fields which are not easily available (the main point that motivates this method). The fact that the method uses images and anatomical labels in training while only the images are used in testing is closely related to _learning with privileged information_. One possibile formulation of the weak supervision is _multiple instance learning_, but this may be beyond the scope of this tutorial. Instead, the initial work [Hu et al ISBI2018][Hu2018b] explored an intuitive idea to consider a two-class classification problem, where the abstract concept of correspondence was used. The two voxels, warped and fixed, at the same imaging coordinate belong to either correspondent or not correspondent. In this case, the segmentations of different types of anatomical regions become noise-corrupted _data_ of these correspondence, that is, being correspondent if both are foreground 1s or both are background 0s, not correspondent otherwise. A well-defined cross-entropy was used to measure the overall classification loss. 

The main problem with the two-class formulation is weighting. The cross-entropy assumes no difference between voxel locations that are nearer to boundaries and those are not. It does not distinguish difference between foreground and background, which can be largely altered by imaging parameters (such as acquired fields-of-view). It treats different types of anatomical regions with different volumes/shapes equally. For example, a voxel in the foreground segmentation of the gland should, without any other information, be a _weaker_ correspondence label than that from a foreground voxel from very small landmark, as the latter is a _strong_ predictor for where this voxel should move to, although it helps very little to indicate the displacement field everywhere else in the background. This is why some heavy heuristic preprocessing label smoothing was used in the original attempt.

Overlap measures, such as Dice, have an interesting quality to re-weight between these classes, with the changing denominator of Dice (as overlap changes) acting as a "dynamic" weighting mechanism. Therefore, it has been adopted in medical image segmentation tasks with consistently superior performance (after all, it is the measure for evaluating segmentation). A similar effect has also been observed in the label-driven registration tasks. The multi-scale strategy is to mitigate the practical issues on the landmarks with smaller volumes, so they will still produce non-zero gradients even without overlap during training, as shown in the following picture with different types of landmarks being filtered at different scales:
![alt text](https://github.com/YipengHu/example-data/raw/master/label-reg-demo/media/multiscale.jpg "Multiscale Label Filtering") 

It may be the case that, due to the nature of weakly-supervised learning, different formulations of the loss function, or the combination of the loss and the regularisation, is only a different weighting strategy to "guess" the dense correspondence. Without ground-truth (for training and validation), this will inherently be dependent on image modalities and applications. Therefore, it may be better to investigate application-specific loss functions (such as better surrogates of the true TREs on regions of interest).

The downside of Dice, however, is that it lacks a clear interpretation of the weak-supervision, leaning towards a general unsupervised learning where any loss function is sensible if it drives the image alignment. The practical difference worthy a name such as "weak supervision" is perhaps that the loss function is not dependent on the image modality, only applied on segmentation labels. This, to certain degree, is closer to traditional feature-based registration method, while the role of neural network is a better way to learn the feature representation. It also reflects the fact that this method, compared with other unsupervised learning, relies on anatomical knowledge in human labelling instead of statistical properties of image-matching (e.g. through image-based similarity measures).

Even with unlimited data pairs, there ought to be physical bounds of the label availability partly due to the underlying imaging process that simply do not produce voxel-level correspondence information and partly due to limited anatomical knowledge. In this case, prior knowledge on, for example, [application-specific physical transformation][Hu2018c] instead of bending energy for instance and combining with other intensity-based similarity might provide further assistance.


**Acknowledgement** 
The author is grateful for a CMIC Platform Fellowship and a Medical Image Analysis Network Knowledge Exchange Project, both funded by UK EPSRC.




[data]: https://github.com/YipengHu/example-data/raw/master/label-reg-demo/data.zip
[code]: https://github.com/YipengHu/label-reg/archive/master.zip

[config_file]: ./config_demo.ini
[loss_file]: ./labelreg/losses.py
[network_file]: ./labelreg/networks.py
[helper_file]: ./labelreg/helpers.py
[util_file]: ./labelreg/utils.py
[inference_file]: ./inference.py
[training_file]: ./training.py
[app_file]: ./labelreg/apps.py

[Hu2018a]: https://doi.org/10.1016/j.media.2018.07.002
[Hu2018b]: https://arxiv.org/abs/1711.01666
[Hu2018c]: https://arxiv.org/abs/1805.10665
[TheDeepLearningBook]: http://www.deeplearningbook.org/

[niftynet]: http://niftynet.io/
[nibabel]: http://nipy.org/nibabel/
[tensorflow_install]: https://www.tensorflow.org/install/

