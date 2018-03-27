
import numpy as np
import nibabel as nib
import config
import os
import random


def get_data_readers(dir_image, dir_label):
    reader_image = DataReader(dir_image)
    reader_label = DataReader(dir_label)
    if not (reader_image.num_data == reader_label.num_data):
        raise Exception('Unequal num_data between images and labels!')
    if not (reader_image.data_shape == reader_label.data_shape):
        raise Exception('Unequal data_shape between images and labels!')
    return reader_image, reader_label


class DataReader:

    def __init__(self, dir_name):
        self.dir_name = dir_name
        self.files = os.listdir(dir_name)
        self.files.sort()
        self.num_data = len(self.files)

        self.file_objects = [nib.load(os.path.join(dir_name, self.files[i])) for i in range(self.num_data)]
        self.num_labels = [self.file_objects[i].shape[3] if len(self.file_objects[i].shape) == 4
                           else 1
                           for i in range(self.num_data)]

        self.data_shape = list(self.file_objects[0].shape[0:3])

    def get_num_labels(self, case_indices):
        return [self.num_labels[i] for i in case_indices]

    def get_data(self, case_indices, label_indices=None):
        # todo: check the supplied label_indices smaller than num_labels
        if label_indices is None:  # e.g. images only
            data = [np.asarray(self.file_objects[i].dataobj) for i in case_indices]  # np.asarray(proxy_img.dataobj)
        else:
            data = [self.file_objects[i].dataobj[..., j] if self.num_labels[i] > 1
                    else np.asarray(self.file_objects[i].dataobj)
                    for (i, j) in zip(case_indices, label_indices)]
        return np.expand_dims(np.stack(data, axis=0), axis=4)







def random_transform_generator(batch_size, cornerScale=.1):
    offsets = np.tile([[[1.,1.,1.],[1.,1.,-1.],[1.,-1.,1.],[-1.,1.,1.]]],[batch_size,1,1])*np.random.uniform(0,cornerScale,[batch_size,4,3])
    newCorners = np.transpose(np.concatenate((np.tile([[[-1.,-1.,-1.],[-1.,-1.,1.],[-1.,1.,-1.],[1.,-1.,-1.]]],[batch_size,1,1])+offsets,np.ones([batch_size,4,1])),2),[0,1,2]) # O = T I
    srcCorners = np.tile(np.transpose([[[-1.,-1.,-1.,1.],[-1.,-1.,1.,1.],[-1.,1.,-1.,1.],[1.,-1.,-1.,1.]]],[0,1,2]),[batch_size,1,1])
    transforms = np.array([np.linalg.lstsq(srcCorners[k], newCorners[k])[0] for k in range(srcCorners.shape[0])])
    # transforms = transforms*np.concatenate((np.ones([batch_size,1,2]),(-1)**np.random.randint(0,2,[batch_size,1,1]),np.ones([batch_size,1,1])),2) # random LR flip
    transforms = np.reshape(np.transpose(transforms[:][:,:][:,:,:3],[0,2,1]),[-1,1,12])
    return transforms


def initial_transform_generator(batch_size):
    identity = identity_transform_vector()
    transforms = np.reshape(np.tile(identity, batch_size), [batch_size, 1, 12])
    return transforms


def identity_transform_vector():
    identity = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]])
    return identity.flatten()


def get_reference_grid(grid_size):
    return np.stack(np.meshgrid(np.arange(grid_size[0]), np.arange(grid_size[1]), np.arange(grid_size[2]), indexing='ij'), axis=3)


def get_smoothing_kernel(sigma):
    if sigma > 0:  # gaussian
        tail = int(sigma * 2)  # tail = int(sigma*3)
        x, y, z = np.mgrid[-tail:tail+1, -tail:tail+1, -tail:tail+1]
        g = np.exp(-0.5*(x**2+y**2++z**2)/sigma**2)
        return g / g.sum()
    elif sigma < 0:  # bspline
        # TODO: add the b-spline kernel here
        return


def dataset_switcher(dataset_name):  # ['us', '800', '35-3']

    if dataset_name == 'motion_5':
        image = 'sims_motion.h5'
        label = 'glands_motion.h5'
    elif dataset_name == 'motion_mini':
        image = 'sims_motion_mini.h5'
        label = 'glands_motion_mini.h5'
    elif dataset_name[1] == '0':  # dataset_name=['us', '0', '35-3'] - for compatibility
        label = dataset_name[0] + '_labels_post' + dataset_name[2] + '.h5'
        if dataset_name[0] == 'us':
            image = 'us_images_vd1fov110.h5'
        elif dataset_name[0] == 'mr':
            image = 'mr_images_vd1.h5'
    else:
        label = dataset_name[0] + '_labels_resampled' + dataset_name[1] + '_post' + dataset_name[2] + '.h5'
        image = dataset_name[0] + '_images_resampled' + dataset_name[1] + '.h5'

    norm_folder = 'Scratch/data/mrusv2/normalised/'
    label = os.path.join(os.environ['HOME'], norm_folder, label)
    image = os.path.join(os.environ['HOME'], norm_folder, image)

    fid = h5py.File(image)  # has to be image
    if dataset_name[0:6] == 'motion':
        vol_size = [int(fid['sim0000000'].shape[i]) for i in [0, 1, 2]]
    else:
        vol_size = [int(fid['case000000'].shape[i]) for i in [0, 1, 2]]
    dataset_size = len(fid)
    return image, label, vol_size, dataset_size


class DataFeeder:
    def __init__(self, fn_image, fn_label, fn_mask=''):
        self.fn_image = fn_image
        self.id_image = h5py.File(self.fn_image, 'r')
        self.fn_label = fn_label
        self.id_label = h5py.File(self.fn_label, 'r')
        self.num_labels = self.id_label['/num_labels'][0]
        self.num_important = self.id_label['/num_important'][0]
        self.fn_mask = fn_mask
        if len(fn_mask) > 0:
            self.id_mask = h5py.File(self.fn_mask, 'r')
        else:
            self.id_mask = []

    def get_image_batch(self, case_indices):
        group_names = ['/case%06d' % i for i in case_indices]
        return np.expand_dims(np.concatenate([np.expand_dims(self.id_image[i], axis=0) for i in group_names], axis=0), axis=4)

    def get_label_batch(self, case_indices, label_indices=None):
        group_names = ['/case%06d_label%03d' % (i, j) for (i, j) in zip(case_indices, label_indices)]
        return np.expand_dims(np.concatenate([np.expand_dims(self.id_label[i], axis=0) for i in group_names], axis=0), axis=4)

    def get_binary_batch(self, case_indices, label_indices=None):
        group_names = ['/case%06d_bin%03d' % (i, j) for (i, j) in zip(case_indices, label_indices)]
        return np.expand_dims(np.concatenate([np.expand_dims(self.id_label[i], axis=0) for i in group_names], axis=0), axis=4)

    def get_mask_batch(self, case_indices):
        group_names = ['/case%06d' % i for i in case_indices]
        return np.expand_dims(np.concatenate([np.expand_dims(self.id_mask[i], axis=0) for i in group_names], axis=0), axis=4)


class MotionDataFeeder:
    def __init__(self, fn_sim, fn_bin):
        self.fn_sim = fn_sim
        self.id_sim = h5py.File(self.fn_sim, 'r')
        self.fn_bin = fn_bin
        self.id_bin = h5py.File(self.fn_bin, 'r')

    # data in original reverse order [z,x,y] <=> [i,j,k]
    # Ds - original: [1,2,3] <=> [x,y,z] <=> [j,k,i] => tf: [i,j,k] <=> [3,1,2]
    # the above may be pre-computed
    def get_sim_batch(self, case_indices):
        group_names = ['/sim%07d' % i for i in case_indices]
        return np.concatenate([np.expand_dims(self.id_sim[i], axis=0) for i in group_names], axis=0)  # np.concatenate([np.expand_dims(np.transpose(self.id_sim[i][[2,0,1],:,:,:], axis=[1,2,3,0]), axis=0) for i in group_names], axis=0)

    def get_bin_batch(self, case_indices):
        group_names = ['/bin%07d' % i for i in case_indices]
        return np.expand_dims(np.concatenate([np.expand_dims(self.id_bin[i], axis=0) for i in group_names], axis=0), axis=4)


def get_hierarchy_sizes(size, num):
    return [[int(size[i] / 2**j) for i in range(len(size))] for j in range(num)]


def get_padded_shape(size, stride, type='same'):
    return [int(np.ceil(size[i] / stride)) for i in range(len(size))]


def setup_cross_validation_simple(totalDataSize, num_fold, idx_fold, miniBatchSize, num_important):

    foldSize = int(totalDataSize / num_fold)
    dataIndices = [i for i in range(totalDataSize)]
    # random.shuffle(dataIndices)  # shuffle once
    sortedCaseIndices = [1,2]  # pseudo randomness for testing generalisation ability for only a few folds
    while any([(sortedCaseIndices[i + 1] - sortedCaseIndices[i]) == 1 for i in range(len(sortedCaseIndices) - 1)]):
        random.shuffle(dataIndices)
        sortedCaseIndices = [dataIndices[i] for j in range(num_fold) for i in range(foldSize * j, foldSize * (j + 1))]
    testCaseIndices = [dataIndices[i] for i in range(foldSize * idx_fold, foldSize * (idx_fold + 1))]

    # grouping the remainders to test
    remainder = totalDataSize % num_fold
    if (remainder != 0) & (idx_fold < remainder):
        testCaseIndices.append(dataIndices[totalDataSize - remainder + idx_fold])
    num_cases_test = len(testCaseIndices)
    # pre-compute indices for inference using minibatches
    testIndices = []
    label_indices_test = []
    for idx in testCaseIndices:
        testIndices += [idx] * num_important[idx]
        label_indices_test += [i for i in range(num_important[idx])]
    num_labels_test = len(testIndices)
    # padding
    remainder_test = len(testIndices) % miniBatchSize
    if remainder_test > 0:
        testIndices += [testIndices[i] for i in range(miniBatchSize - remainder_test)]
        label_indices_test += [label_indices_test[i] for i in range(miniBatchSize - remainder_test)]
    num_miniBatch_test = int(len(testIndices) / miniBatchSize)

    # make sure test are excluded in train
    trainIndices = list(set(dataIndices) - set(testCaseIndices))
    random.shuffle(trainIndices)
    trainSize = len(trainIndices)
    num_miniBatch = int(trainSize / miniBatchSize)

    return testCaseIndices, testIndices, label_indices_test, num_cases_test, num_labels_test, num_miniBatch_test, trainIndices, num_miniBatch