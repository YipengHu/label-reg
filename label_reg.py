import tensorflow as tf
import config

import helper
import util as tfops
import network as net
import numpy as np


# data sets
h5fn_image_target, h5fn_label_target, size_target, totalDataSize = helpers.dataset_switcher(dataset_name_target)
h5fn_image_moving, h5fn_label_moving, size_moving, totalDataSize2 = helpers.dataset_switcher(dataset_name_moving)




# data feeding
dataset_name_target = ['us', '800', '3']
dataset_name_moving = ['mr', '800', '3']

# local-net parameters
loss_scales = [loss_scales[i]/scale_per_voxel for i in range(len(loss_scales))]
mixture_sigmas = [mixture_sigmas[i]/scale_per_voxel for i in range(len(mixture_sigmas))]

# other options
initialiser_local = tf.random_normal_initializer(0, initial_std_local)  # tf.contrib.layers.xavier_initializer()
initialiser_global = tf.random_normal_initializer(0, initial_std_global)  # tf.contrib.layers.xavier_initializer()
initialiser_default = tf.contrib.layers.xavier_initializer()
transform_initial = tfhelpers.identity_transform_vector()
strides_none = [1, 1, 1, 1, 1]
strides_down = [1, 2, 2, 2, 1]
k_conv0 = [conv_size_initial, conv_size_initial, conv_size_initial]
k_conv = [3, 3, 3]
k_pool = [1, 2, 2, 2, 1]


if not (totalDataSize == totalDataSize2):
    raise Exception('moving and target data should have the same size!')
feeder_target = tfhelpers.DataFeeder(h5fn_image_target, h5fn_label_target)
feeder_moving = tfhelpers.DataFeeder(h5fn_image_moving, h5fn_label_moving)
size_1 = tfhelpers.get_padded_shape(size_target, 2)
size_2 = tfhelpers.get_padded_shape(size_target, 4)
size_3 = tfhelpers.get_padded_shape(size_target, 8)
size_4 = tfhelpers.get_padded_shape(size_target, 16)
vol_target = size_target[0] * size_target[1] * size_target[2]

# logging
filename_output_info = 'output_info.txt'
dir_output = os.path.join(os.environ['HOME'], 'Scratch/output/labelreg0/', "%f" % time.time())
flag_dir_overwrite = os.path.exists(dir_output)
os.makedirs(dir_output)
fid_output_info = open(os.path.join(dir_output, filename_output_info), 'a')
if flag_dir_overwrite:
    print('\nWARNING: %s existed - files may be overwritten.\n\n' % dir_output)
    print('\nWARNING: %s existed - files may be overwritten.\n\n' % dir_output, flush=True, file=fid_output_info)


# now sanity-check the parameters
if network_type == 'global-only':
    start_train_local = totalIterations+1
    start_train_composite = totalIterations+1

if ddf_composing:
    if not ((ddf_scales[0]==[0,1,2,3,4]) & (len(ddf_scales)==1)):
        raise Exception('DDF scales is not currently supported by composing method!')


# --- set up the cross-validation ---
int_seed = 1
random.seed(int_seed)
tf.set_random_seed(int_seed)
num_important = feeder_target.num_important
testCaseIndices, testIndices, label_indices_test, num_cases_test, num_labels_test, num_miniBatch_test, trainIndices, num_miniBatch \
    = tfhelpers.setup_cross_validation_simple(totalDataSize, num_fold, idx_fold, miniBatchSize, num_important)

# pre-computing for graph
transform_identity = tfhelpers.initial_transform_generator(miniBatchSize)
grid_reference = tftools.get_reference_grid(size_target)
grid_moving = tf.stack([tftools.get_reference_grid(size_moving)]*miniBatchSize, axis=0)  # tf.expand_dims(tftools.get_reference_grid(size_moving), axis=0)


# information
print('- Algorithm Summary (fly) --------', flush=True, file=fid_output_info)

print('current_time: %s' % time.asctime(time.gmtime()), flush=True, file=fid_output_info)
print('flag_dir_overwrite: %s' % flag_dir_overwrite, flush=True, file=fid_output_info)

print('lambda_decay: %s' % lambda_decay, flush=True, file=fid_output_info)
print('regulariser_type_local: %s' % regulariser_type_local, flush=True, file=fid_output_info)
print('lambda_local: %s' % lambda_local, flush=True, file=fid_output_info)

print('totalIterations: %s' % totalIterations, flush=True, file=fid_output_info)
print('network_type: %s' % network_type, flush=True, file=fid_output_info)
print('loss_type: %s' % loss_type, flush=True, file=fid_output_info)
print('loss_scales: %s' % loss_scales, flush=True, file=fid_output_info)
print('label_pre_smooth: %s' % label_pre_smooth, flush=True, file=fid_output_info)
print('mixture_sigmas: %s' % mixture_sigmas, flush=True, file=fid_output_info)
print('mixture_kernel: %s' % mixture_kernel, flush=True, file=fid_output_info)
print('learning_rate: %s' % learning_rate, flush=True, file=fid_output_info)
print('start_train_local: %s' % start_train_local, flush=True, file=fid_output_info)
print('start_train_composite: %s' % start_train_composite, flush=True, file=fid_output_info)

print('ddf_scales: %s' % ddf_scales, flush=True, file=fid_output_info)
print('ddf_scale_iterations: %s' % ddf_scale_iterations, flush=True, file=fid_output_info)
print('ddf_composing: %s' % ddf_composing, flush=True, file=fid_output_info)

print('dataset_name_target: %s' % dataset_name_target, flush=True, file=fid_output_info)
print('dataset_name_moving: %s' % dataset_name_moving, flush=True, file=fid_output_info)
print('start_feed_type: %s' % start_feed_type, flush=True, file=fid_output_info)
print('start_feed_multiple: %s' % start_feed_multiple, flush=True, file=fid_output_info)
print('start_feed_all: %s' % start_feed_all, flush=True, file=fid_output_info)
print('feed_importance_sampling: %s' % feed_importance_sampling, flush=True, file=fid_output_info)

print('num_channel_initial: %s' % num_channel_initial, flush=True, file=fid_output_info)
print('conv_size_initial: %s' % conv_size_initial, flush=True, file=fid_output_info)
print('num_channel_initial_global: %s' % num_channel_initial_global, flush=True, file=fid_output_info)

print('miniBatchSize: %s' % miniBatchSize, flush=True, file=fid_output_info)
print('initial_bias_global: %s' % initial_bias_global, flush=True, file=fid_output_info)
print('initial_bias_local: %s' % initial_bias_local, flush=True, file=fid_output_info)
print('initial_std_global: %s' % initial_std_global, flush=True, file=fid_output_info)
print('initial_std_local: %s' % initial_std_local, flush=True, file=fid_output_info)
print('int_seed: %s' % int_seed, flush=True, file=fid_output_info)
print('num_fold: %s' % num_fold, flush=True, file=fid_output_info)
print('idx_fold: %s' % idx_fold, flush=True, file=fid_output_info)

print('- End of Algorithm Summary --------', flush=True, file=fid_output_info)

if log_num_shuffle:
    print('trainDataIndices: %s' % trainIndices, flush=True, file=fid_output_info)


# --- global-net ---
nc0_g = num_channel_initial_global
W0c_g = tf.get_variable("W0c_g", shape=k_conv0 + [2, nc0_g], initializer=initialiser_default)
W0r1_g = tf.get_variable("W0r1_g", shape=k_conv + [nc0_g, nc0_g], initializer=initialiser_default)
W0r2_g = tf.get_variable("W0r2_g", shape=k_conv + [nc0_g, nc0_g], initializer=initialiser_default)
vars_global = [W0c_g, W0r1_g, W0r2_g]
nc1_g = nc0_g * 2
W1c_g = tf.get_variable("W1c_g", shape=k_conv + [nc0_g, nc1_g], initializer=initialiser_default)
W1r1_g = tf.get_variable("W1r1_g", shape=k_conv + [nc1_g, nc1_g], initializer=initialiser_default)
W1r2_g = tf.get_variable("W1r2_g", shape=k_conv + [nc1_g, nc1_g], initializer=initialiser_default)
vars_global += [W1c_g, W1r1_g, W1r2_g]
nc2_g = nc1_g * 2
W2c_g = tf.get_variable("W2c_g", shape=k_conv + [nc1_g, nc2_g], initializer=initialiser_default)
W2r1_g = tf.get_variable("W2r1_g", shape=k_conv + [nc2_g, nc2_g], initializer=initialiser_default)
W2r2_g = tf.get_variable("W2r2_g", shape=k_conv + [nc2_g, nc2_g], initializer=initialiser_default)
vars_global += [W2c_g, W2r1_g, W2r2_g]
nc3_g = nc2_g * 2
W3c_g = tf.get_variable("W3c_g", shape=k_conv + [nc2_g, nc3_g], initializer=initialiser_default)
W3r1_g = tf.get_variable("W3r1_g", shape=k_conv + [nc3_g, nc3_g], initializer=initialiser_default)
W3r2_g = tf.get_variable("W3r2_g", shape=k_conv + [nc3_g, nc3_g], initializer=initialiser_default)
vars_global += [W3c_g, W3r1_g, W3r2_g]
ncD_g = nc3_g * 2  # deep layer
WD_g = tf.get_variable("WD_g", shape=k_conv + [nc3_g, ncD_g], initializer=initialiser_default)
vars_global += [WD_g]
nfD = size_4[0]*size_4[1]*size_4[2]*ncD_g
W_global = tf.get_variable("W_global", shape=[nfD, 12], initializer=initialiser_global)
b_global = tf.get_variable("b_global", shape=[1, 12], initializer=tf.constant_initializer(
    transform_initial + initial_bias_global))  # tf.Variable(tf.squeeze(tf.to_float(transform_identity+initial_bias), axis=1), name='b_global')
vars_global += [W_global]  # no bias term


def global_net(input_image_pair):  # input_image_pair = tf.concat([tftools.resize_volume(movingImage0, size_target), targetImage0], axis=4)
    h0c_g = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(input_image_pair, W0c_g, strides_none,"SAME")))
    h0r1_g = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h0c_g, W0r1_g, strides_none, "SAME")))
    h0r2_g = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h0r1_g, W0r2_g, strides_none, "SAME")) + h0c_g)
    h0_g = tf.nn.max_pool3d(h0r2_g, k_pool, strides_down, padding="SAME")
    h1c_g = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h0_g, W1c_g, strides_none, "SAME")))
    h1r1_g = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h1c_g, W1r1_g, strides_none, "SAME")))
    h1r2_g = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h1r1_g, W1r2_g, strides_none, "SAME")) + h1c_g)
    h1_g = tf.nn.max_pool3d(h1r2_g, k_pool, strides_down, padding="SAME")
    h2c_g = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h1_g, W2c_g, strides_none, "SAME")))
    h2r1_g = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h2c_g, W2r1_g, strides_none, "SAME")))
    h2r2_g = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h2r1_g, W2r2_g, strides_none, "SAME")) + h2c_g)
    h2_g = tf.nn.max_pool3d(h2r2_g, k_pool, strides_down, padding="SAME")
    h3c_g = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h2_g, W3c_g, strides_none, "SAME")))
    h3r1_g = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h3c_g, W3r1_g, strides_none, "SAME")))
    h3r2_g = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h3r1_g, W3r2_g, strides_none, "SAME")) + h3c_g)
    h3_g = tf.nn.max_pool3d(h3r2_g, k_pool, strides_down, padding="SAME")
    hD_g = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h3_g, WD_g, strides_none, "SAME")))
    return tf.matmul(tf.reshape(hD_g, [miniBatchSize, -1]), W_global) + b_global


# --- local-net ---
nc0 = num_channel_initial
W0c = tf.get_variable("W0c", shape=k_conv0+[2, nc0], initializer=initialiser_default)
W0r1 = tf.get_variable("W0r1", shape=k_conv+[nc0, nc0], initializer=initialiser_default)
W0r2 = tf.get_variable("W0r2", shape=k_conv+[nc0, nc0], initializer=initialiser_default)
vars_local = [W0c, W0r1, W0r2]
nc1 = nc0*2
W1c = tf.get_variable("W1c", shape=k_conv+[nc0, nc1], initializer=initialiser_default)
W1r1 = tf.get_variable("W1r1", shape=k_conv+[nc1, nc1], initializer=initialiser_default)
W1r2 = tf.get_variable("W1r2", shape=k_conv+[nc1, nc1], initializer=initialiser_default)
vars_local += [W1c, W1r1, W1r2]
nc2 = nc1*2
W2c = tf.get_variable("W2c", shape=k_conv+[nc1, nc2], initializer=initialiser_default)
W2r1 = tf.get_variable("W2r1", shape=k_conv+[nc2, nc2], initializer=initialiser_default)
W2r2 = tf.get_variable("W2r2", shape=k_conv+[nc2, nc2], initializer=initialiser_default)
vars_local += [W2c, W2r1, W2r2]
nc3 = nc2*2
W3c = tf.get_variable("W3c", shape=k_conv+[nc2, nc3], initializer=initialiser_default)
W3r1 = tf.get_variable("W3r1", shape=k_conv+[nc3, nc3], initializer=initialiser_default)
W3r2 = tf.get_variable("W3r2", shape=k_conv+[nc3, nc3], initializer=initialiser_default)
vars_local += [W3c, W3r1, W3r2]
nc4 = nc3*2  # deep layer
WD = tf.get_variable("WD", shape=k_conv+[nc3, nc4], initializer=initialiser_default)
vars_local += [WD]
W3_c = tf.get_variable("W3_c", shape=k_conv+[nc3, nc4], initializer=initialiser_default)
W3_r1 = tf.get_variable("W3_r1", shape=k_conv+[nc3, nc3], initializer=initialiser_default)
W3_r2 = tf.get_variable("W3_r2", shape=k_conv+[nc3, nc3], initializer=initialiser_default)
vars_local += [W3_c, W3_r1, W3_r2]
W2_c = tf.get_variable("W2_c", shape=k_conv+[nc2, nc3], initializer=initialiser_default)
W2_r1 = tf.get_variable("W2_r1", shape=k_conv+[nc2, nc2], initializer=initialiser_default)
W2_r2 = tf.get_variable("W2_r2", shape=k_conv+[nc2, nc2], initializer=initialiser_default)
vars_local += [W2_c, W2_r1, W2_r2]
W1_c = tf.get_variable("W1_c", shape=k_conv+[nc1, nc2], initializer=initialiser_default)
W1_r1 = tf.get_variable("W1_r1", shape=k_conv+[nc1, nc1], initializer=initialiser_default)
W1_r2 = tf.get_variable("W1_r2", shape=k_conv+[nc1, nc1], initializer=initialiser_default)
vars_local += [W1_c, W1_r1, W1_r2]
W0_c = tf.get_variable("W0_c", shape=k_conv+[nc0, nc1], initializer=initialiser_default)
W0_r1 = tf.get_variable("W0_r1", shape=k_conv+[nc0, nc0], initializer=initialiser_default)
W0_r2 = tf.get_variable("W0_r2", shape=k_conv+[nc0, nc0], initializer=initialiser_default)
vars_local += [W0_c, W0_r1, W0_r2]
# output
W_ddf = [tf.get_variable("W_ddf0", shape=k_conv+[nc0, 3], initializer=initialiser_local),
         tf.get_variable("W_ddf1", shape=k_conv+[nc1, 3], initializer=initialiser_local),
         tf.get_variable("W_ddf2", shape=k_conv+[nc2, 3], initializer=initialiser_local),
         tf.get_variable("W_ddf3", shape=k_conv+[nc3, 3], initializer=initialiser_local),
         tf.get_variable("W_ddf4", shape=k_conv+[nc4, 3], initializer=initialiser_local)]
b_ddf = [tf.get_variable("b_ddf0", shape=[3], initializer=tf.constant_initializer(initial_bias_local)),
         tf.get_variable("b_ddf1", shape=[3], initializer=tf.constant_initializer(initial_bias_local)),
         tf.get_variable("b_ddf2", shape=[3], initializer=tf.constant_initializer(initial_bias_local)),
         tf.get_variable("b_ddf3", shape=[3], initializer=tf.constant_initializer(initial_bias_local)),
         tf.get_variable("b_ddf4", shape=[3], initializer=tf.constant_initializer(initial_bias_local))]
# vars_local += [W_ddf4, W_ddf3, W_ddf2, W_ddf1, W_ddf0]  # no bias terms


def local_net(input_image_pair):  # for debug: input_image_pair=tf.concat([tftools.resize_volume(movingImage0, size_target), targetImage0], axis=4)
    # down-sampling
    h0c = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(input_image_pair, W0c, strides_none, "SAME")))
    h0r1 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h0c, W0r1, strides_none, "SAME")))
    h0r2 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h0r1, W0r2, strides_none, "SAME")) + h0c)
    h0 = tf.nn.max_pool3d(h0r2, k_pool, strides_down, padding="SAME")
    h1c = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h0, W1c, strides_none, "SAME")))
    h1r1 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h1c, W1r1, strides_none, "SAME")))
    h1r2 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h1r1, W1r2, strides_none, "SAME")) + h1c)
    h1 = tf.nn.max_pool3d(h1r2, k_pool, strides_down, padding="SAME")
    h2c = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h1, W2c, strides_none, "SAME")))
    h2r1 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h2c, W2r1, strides_none, "SAME")))
    h2r2 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h2r1, W2r2, strides_none, "SAME")) + h2c)
    h2 = tf.nn.max_pool3d(h2r2, k_pool, strides_down, padding="SAME")
    h3c = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h2, W3c, strides_none, "SAME")))
    h3r1 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h3c, W3r1, strides_none, "SAME")))
    h3r2 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h3r1, W3r2, strides_none, "SAME")) + h3c)
    h3 = tf.nn.max_pool3d(h3r2, k_pool, strides_down, padding="SAME")
    # deep
    h4 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h3, WD, strides_none, "SAME")))
    # up-sampling
    h3_c = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d_transpose(h4, W3_c, h3c.get_shape(), strides_down, "SAME")))
    h3_c += tftools.additive_up_sampling(h4, size_3)
    h3_r1 = tf.add(h3_c, h3c)
    h3_r2 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h3_r1, W3_r1, strides_none, "SAME")))
    h3_ = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h3_r2, W3_r2, strides_none, "SAME")) + h3_r1)
    h2_c = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d_transpose(h3_, W2_c, h2c.get_shape(), strides_down, "SAME")))
    h2_c += tftools.additive_up_sampling(h3_, size_2)
    h2_r1 = tf.add(h2_c, h2c)
    h2_r2 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h2_r1, W2_r1, strides_none, "SAME")))
    h2_ = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h2_r2, W2_r2, strides_none, "SAME")) + h2_r1)
    h1_c = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d_transpose(h2_, W1_c, h1c.get_shape(), strides_down, "SAME")))
    h1_c += tftools.additive_up_sampling(h2_, size_1)
    h1_r1 = tf.add(h1_c, h1c)
    h1_r2 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h1_r1, W1_r1, strides_none, "SAME")))
    h1_ = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h1_r2, W1_r2, strides_none, "SAME")) + h1_r1)
    h0_c = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d_transpose(h1_, W0_c, h0c.get_shape(), strides_down, "SAME")))
    h0_c += tftools.additive_up_sampling(h1_, size_target)
    h0_r1 = tf.add(h0_c, h0c)
    h0_r2 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h0_r1, W0_r1, strides_none, "SAME")))
    h0_ = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h0_r2, W0_r2, strides_none, "SAME")) + h0_r1)

    if network_type == 'multi-scale':
        if ddf_composing:  # it is not suitable for composing with global-net
            ddf4 = tftools.resize_volume(tf.nn.conv3d(h4, W_ddf[4], strides_none, "SAME") + b_ddf[4], size_target)
            ddf3 = tftools.resize_volume(tf.nn.conv3d(h3_, W_ddf[3], strides_none, "SAME") + b_ddf[3], size_target)
            ddf2 = tftools.resize_volume(tf.nn.conv3d(h2_, W_ddf[2], strides_none, "SAME") + b_ddf[2], size_target)
            ddf1 = tftools.resize_volume(tf.nn.conv3d(h1_, W_ddf[1], strides_none, "SAME") + b_ddf[1], size_target)
            ddf0 = tf.nn.conv3d(h0_, W_ddf[0], strides_none, "SAME") + b_ddf[0]

            grid0 = tftools.resample_linear(grid_moving, grid_reference + ddf0)  # initialise with grid_reference
            grid1 = tftools.resample_linear(grid_moving, grid0 + ddf1)
            grid2 = tftools.resample_linear(grid_moving, grid1 + ddf2)
            grid3 = tftools.resample_linear(grid_moving, grid2 + ddf3)
            grid4 = tftools.resample_linear(grid_moving, grid3 + ddf4)
            ddf = grid4 - grid_reference
        else:  # summation only
            ddf = tftools.resize_volume(tf.nn.conv3d(h4, W_ddf[4], strides_none, "SAME") + b_ddf[4], size_target) + \
                  tftools.resize_volume(tf.nn.conv3d(h3_, W_ddf[3], strides_none, "SAME") + b_ddf[3], size_target) + \
                  tftools.resize_volume(tf.nn.conv3d(h2_, W_ddf[2], strides_none, "SAME") + b_ddf[2], size_target) + \
                  tftools.resize_volume(tf.nn.conv3d(h1_, W_ddf[1], strides_none, "SAME") + b_ddf[1], size_target) + \
                  tf.nn.conv3d(h0_, W_ddf[0], strides_none, "SAME") + b_ddf[0]
    else:
        ddf = tf.nn.conv3d(h0_, W_ddf[0], strides_none, "SAME") + b_ddf[0]

    return ddf


# building the computational graph
movingImage_ph = tf.placeholder(tf.float32, [miniBatchSize]+size_moving+[1])
targetImage_ph = tf.placeholder(tf.float32, [miniBatchSize]+size_target+[1])
movingLabel_ph = tf.placeholder(tf.float32, [miniBatchSize]+size_moving+[1])
targetLabel_ph = tf.placeholder(tf.float32, [miniBatchSize]+size_target+[1])
movingTransform_ph = tf.placeholder(tf.float32, [miniBatchSize]+[1, 12])
targetTransform_ph = tf.placeholder(tf.float32, [miniBatchSize]+[1, 12])
# keep_prob_ph = tf.placeholder(tf.float32)

# random spatial transform - do not rescaling moving here
movingImage0 = tftools.random_transform1(movingImage_ph, movingTransform_ph, size_moving)
targetImage0 = tftools.random_transform1(targetImage_ph, targetTransform_ph, size_target)

energy = tf.constant(0.0)
if network_type == 'global-only':
    theta = global_net(tf.concat([tftools.resize_volume(movingImage0, size_target), targetImage0], axis=4))
    grid_sample_global = tftools.warp_grid(grid_reference, theta)
    displacement_local = tf.constant(0.0)
else:
    if network_type == 'multi-scale':
        grid_sample_global = grid_reference
        displacement_local = local_net(tf.concat([tftools.resize_volume(movingImage0, size_target), targetImage0], axis=4))
    elif network_type == 'two-stream':
        theta = global_net(tf.concat([tftools.resize_volume(movingImage0, size_target), targetImage0], axis=4))
        grid_sample_global = tftools.warp_grid(grid_reference, theta)
        displacement_local = local_net(tf.concat([tftools.resize_volume(movingImage0, size_target), targetImage0], axis=4))
    elif network_type == 'pre-warp':
        theta = global_net(tf.concat([tftools.resize_volume(movingImage0, size_target), targetImage0], axis=4))
        grid_sample_global = tftools.warp_grid(grid_reference, theta)
        displacement_local = local_net(tf.concat([tftools.resample_linear(movingImage0, grid_sample_global), targetImage0], axis=4))
    else:
        raise Exception('Not recognised network!')
    # local-net only
    if lambda_local:
        if regulariser_type_local == 'bending':
            energy = tftools.compute_bending_energy(displacement_local)
        elif regulariser_type_local == 'gradient-l2':
            energy = tftools.compute_gradient_norm(displacement_local)
        elif regulariser_type_local == 'gradient-l1':
            energy = tftools.compute_gradient_norm(displacement_local, True)
        else:
            raise Exception('Not recognised local regulariser!')
        energy = tf.reduce_mean(energy)
        tf.add_to_collection('loss', energy*lambda_local)


# label correspondence
movingLabel0 = tftools.random_transform1(movingLabel_ph, movingTransform_ph, size_moving)
targetLabel0 = tftools.random_transform1(targetLabel_ph, targetTransform_ph, size_target)
if label_pre_smooth:
    movingLabel = tftools.mixture_filter3d(movingLabel_ph, mixture_sigmas, mixture_kernel)
    targetLabel = tftools.mixture_filter3d(targetLabel_ph, mixture_sigmas, mixture_kernel)
    movingLabel = tftools.random_transform1(movingLabel, movingTransform_ph, size_moving)
    targetLabel = tftools.random_transform1(targetLabel, targetTransform_ph, size_target)

movingLabel_warped0 = tftools.resample_linear(movingLabel0, grid_sample_global + displacement_local)  # eval+post
if label_pre_smooth:
    movingLabel_warped = tftools.resample_linear(movingLabel, grid_sample_global + displacement_local)
else:
    movingLabel_warped = tftools.mixture_filter3d(movingLabel_warped0, mixture_sigmas, mixture_kernel)
    targetLabel = tftools.mixture_filter3d(targetLabel0, mixture_sigmas, mixture_kernel)  # recompute due to data augmentation

label_loss_batch = tftools.multi_scale_loss(targetLabel, movingLabel_warped, loss_type, loss_scales)
label_loss = tf.reduce_mean(label_loss_batch)
tf.add_to_collection('loss', tf.reduce_mean(label_loss))

# weight decay
if lambda_decay:
    for i in range(len(vars_local)):
        tf.add_to_collection('loss', tf.nn.l2_loss(vars_local[i])*lambda_decay)
    for i in range(len(vars_global)):
        tf.add_to_collection('loss', tf.nn.l2_loss(vars_global[i])*lambda_decay)
    for i in range(len(W_ddf)):
        tf.add_to_collection('loss', tf.nn.l2_loss(W_ddf[i]*lambda_decay))

# loss collections
loss = tf.add_n(tf.get_collection('loss'))

# train options
if (network_type == 'global-only') | (network_type == 'pre-warp') | (network_type == 'two-stream'):
    train_op_global = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=vars_global+[b_global])
if (network_type == 'pre-warp') | (network_type == 'two-stream'):
    train_op_composite = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=vars_global+vars_local+[b_global, W_ddf[0], b_ddf[0]])
    train_op_local = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=vars_local+[W_ddf[0], b_ddf[0]])  # for training stages
if network_type == 'multi-scale':
    train_op_scales = [tf.train.AdamOptimizer(learning_rate).minimize(
        loss, var_list=vars_local + [W_ddf[i] for i in ddf_scales[j]] + [b_ddf[i] for i in ddf_scales[j]])
        for j in range(len(ddf_scales))]


# utility nodes - before filtering
# movingLabel_warped0 = tftools.resample_linear(movingLabel0, grid_sample_global + displacement_local)
dice, movingVol, targetVol = tftools.compute_dice(movingLabel_warped0, targetLabel0)
dist = tftools.compute_centroid_distance(movingLabel_warped0, targetLabel0, grid_reference)
movingLabel_warped_global = tftools.resample_linear(movingLabel0, grid_sample_global)  # TODO: bug here - needs to be add dimension to grid_sample_global if needed
if (network_type == 'pre-warp') | (network_type == 'two-stream'):
    dice_global, movingVol_global, targetVol_global = tftools.compute_dice(movingLabel_warped_global, targetLabel0)
    dist_global = tftools.compute_centroid_distance(movingLabel_warped_global, targetLabel0, grid_reference)


# training
if memory_fraction == 1:
    sess = tf.Session()
else:
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=memory_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

sess.run(tf.global_variables_initializer())
for step in range(totalIterations):
    current_time = time.asctime(time.gmtime())

    if step in range(0, totalIterations, num_miniBatch):
        random.shuffle(trainIndices)
        if step < num_miniBatch * log_num_shuffle:
            print('trainDataIndices: %s' % trainIndices, flush=True, file=fid_output_info)

    miniBatch_idx = step % num_miniBatch
    miniBatch_indices = trainIndices[miniBatch_idx*miniBatchSize:(miniBatch_idx + 1)*miniBatchSize]
    if step < num_miniBatch * log_num_shuffle:
        print('miniBatch_indices: %s' % miniBatch_indices, flush=True, file=fid_output_info)

    # feeding
    if step < start_feed_multiple:
        if start_feed_type == 1:
            label_indices = [0] * miniBatchSize
        elif start_feed_type == 2:  # experimental using only important without 0
            label_indices = [random.randrange(feeder_moving.num_important[i] - 1) + 1 for i in miniBatch_indices]
    elif step < start_feed_all:  # only important labels
        label_indices = [random.randrange(feeder_moving.num_important[i]) for i in miniBatch_indices]
    else:  # importance sampling: hack - todo: use proper sampling
        if random.random() < feed_importance_sampling:
            label_indices = [random.randrange(feeder_moving.num_important[i]) for i in miniBatch_indices]
        else:
            label_indices = [random.randrange(feeder_moving.num_labels[i]) for i in miniBatch_indices]

    trainFeed = {movingImage_ph: feeder_moving.get_image_batch(miniBatch_indices),
                 targetImage_ph: feeder_target.get_image_batch(miniBatch_indices),
                 movingLabel_ph: feeder_moving.get_binary_batch(miniBatch_indices, label_indices),
                 targetLabel_ph: feeder_target.get_binary_batch(miniBatch_indices, label_indices),
                 movingTransform_ph: tfhelpers.random_transform_generator(miniBatchSize),
                 targetTransform_ph: tfhelpers.random_transform_generator(miniBatchSize)}
    
    # training
    if network_type == 'multi-scale':
        for current_scale in range(len(ddf_scale_iterations)):
            if step <= ddf_scale_iterations[current_scale]:
                break
        sess.run(train_op_scales[current_scale], feed_dict=trainFeed)
    else:
        if step < start_train_local:
            sess.run(train_op_global, feed_dict=trainFeed)
        elif step < start_train_composite:
            sess.run(train_op_local, feed_dict=trainFeed)
        else:
            sess.run(train_op_composite, feed_dict=trainFeed)

    if step in range(0, totalIterations, log_freq_info):
        if network_type == 'multi-scale':
            print('----- Training scales: %s -----' % ddf_scales[current_scale])
            print('----- Training scales: %s -----' % ddf_scales[current_scale], flush=True, file=fid_output_info)

        loss_train, label_loss_train, label_loss_batch_train, energy_train = sess.run([loss, label_loss, label_loss_batch, energy], feed_dict=trainFeed)
        print('Step %d [%s]: loss=%f, label=%f, energy=%f' % (step, current_time, loss_train, label_loss_train, energy_train))
        print('Step %d [%s]: loss=%f, label=%f, energy=%f' % (step, current_time, loss_train, label_loss_train, energy_train), flush=True, file=fid_output_info)

        if flag_debugPrint:
            print('  label_loss_batch_train: %s' % label_loss_batch_train)
            print('  label_loss_batch_train: %s' % label_loss_batch_train, flush=True, file=fid_output_info)
            # trainFeed[movingLabel_ph] = feeder_moving.get_binary_batch(miniBatch_indices, label_indices)
            # trainFeed[targetLabel_ph] = feeder_target.get_binary_batch(miniBatch_indices, label_indices)
            print('  DEBUG-PRINT: miniBatch_indices: %s' % miniBatch_indices)
            print('  DEBUG-PRINT: miniBatch_indices: %s' % miniBatch_indices, flush=True, file=fid_output_info)
            print('  DEBUG-PRINT: label_indices: %s' % label_indices)
            print('  DEBUG-PRINT: label_indices: %s' % label_indices, flush=True, file=fid_output_info)
            dice_train, dist_train, movingVol_train, targetVol_train,  = sess.run([dice, dist, movingVol, targetVol], feed_dict=trainFeed)
            print('  Dice: %s' % dice_train)
            print('  Dice: %s' % dice_train, flush=True, file=fid_output_info)
            print('  Distance: %s' % dist_train)
            print('  Distance: %s' % dist_train, flush=True, file=fid_output_info)
            print('  movingVol: %s' % movingVol_train)
            print('  movingVol: %s' % movingVol_train, flush=True, file=fid_output_info)
            print('  targetVol: %s' % targetVol_train)
            print('  targetVol: %s' % targetVol_train, flush=True, file=fid_output_info)
            if (network_type == 'pre-warp') | (network_type == 'two-stream'):
                dice_global_train, dist_global_train, movingVol_global_train, targetVol_global_train = sess.run(
                    [dice_global, dist_global, movingVol_global, targetVol_global], feed_dict=trainFeed)
                print('  Dice (global): %s' % dice_global_train)
                print('  Dice (global): %s' % dice_global_train, flush=True, file=fid_output_info)
                print('  Distance (global): %s' % dist_global_train)
                print('  Distance (global): %s' % dist_global_train, flush=True, file=fid_output_info)
                print('  movingVol_global: %s' % movingVol_global_train)
                print('  movingVol_global: %s' % movingVol_global_train, flush=True, file=fid_output_info)
                print('  targetVol_global: %s' % targetVol_global_train)
                print('  targetVol_global: %s' % targetVol_global_train, flush=True, file=fid_output_info)
            # dice_batch_train, dist_batch_train = sess.run([dice_batch, dist_batch], feed_dict=trainFeed)
            # print('  DEBUG-PRINT: extras')
            # print('  DEBUG-PRINT: extras', flush=True, file=fid_output_info)
            # print('  dice_batch_train: %s' % dice_batch_train)
            # print('  dice_batch_train: %s' % dice_batch_train, flush=True, file=fid_output_info)
            # print('  dist_batch_train: %s' % dist_batch_train)
            # print('  dist_batch_train: %s' % dist_batch_train, flush=True, file=fid_output_info)
            # pos_label_weight_train = sess.run(pos_label_weight, feed_dict=trainFeed)
            # print('DEBUG-PRINT: pos_label_weight: %s' % pos_label_weight_train)

    # Debug data
    if step in range(log_start_debug, totalIterations, log_freq_debug):

        # file
        if log_latest_debug_data > 0:
            filename_log_data = "debug_data_i%09d.h5" % int((step-log_start_debug)/log_freq_debug % log_latest_debug_data)
        else:
            filename_log_data = "debug_data_i%09d.h5" % step

        fid_debug_data = h5py.File(os.path.join(dir_output, filename_log_data), 'w')
        fid_debug_data.create_dataset('/step/', data=step)

        # --- train ---
        fid_debug_data.create_dataset('/miniBatch_indices/', data=miniBatch_indices)
        fid_debug_data.create_dataset('/label_indices/', data=label_indices)

        # --- choose the variables to save ---
        if flag_debugMore:
            movingImage0_train, targetImage0_train = sess.run([movingImage0, targetImage0], feed_dict=trainFeed)
            fid_debug_data.create_dataset('/movingImage0_train/', movingImage0_train.shape, dtype=movingImage0_train.dtype, data=movingImage0_train)
            fid_debug_data.create_dataset('/targetImage0_train/', targetImage0_train.shape, dtype=targetImage0_train.dtype, data=targetImage0_train)
            movingLabel0_train, targetLabel0_train, targetLabel_train, movingLabel_warped_train = sess.run(
                [movingLabel0, targetLabel0, targetLabel, movingLabel_warped], feed_dict=trainFeed)
            fid_debug_data.create_dataset('/movingLabel0_train/', movingLabel0_train.shape, dtype=movingLabel0_train.dtype, data=movingLabel0_train)
            fid_debug_data.create_dataset('/targetLabel0_train/', targetLabel0_train.shape, dtype=targetLabel0_train.dtype, data=targetLabel0_train)
            fid_debug_data.create_dataset('/targetLabel_train/', targetLabel_train.shape, dtype=targetLabel_train.dtype, data=targetLabel_train)
            fid_debug_data.create_dataset('/movingLabel_warped_train/', movingLabel_warped_train.shape, dtype=movingLabel_warped_train.dtype, data=movingLabel_warped_train)

            displacement_local_train = sess.run(displacement_local, feed_dict=trainFeed)
            fid_debug_data.create_dataset('/displacement_local_train/', displacement_local_train.shape, dtype=displacement_local_train.dtype, data=displacement_local_train)
            if (network_type == 'pre-warp') | (network_type == 'two-stream'):
                grid_sample_global_train = sess.run(grid_sample_global, feed_dict=trainFeed)
                fid_debug_data.create_dataset('/grid_sample_global_train/', grid_sample_global_train.shape, dtype=grid_sample_global_train.dtype, data=grid_sample_global_train)

        # --- test ---
        fid_debug_data.create_dataset('/testIndices/', data=testIndices)
        fid_debug_data.create_dataset('/label_indices_test/', data=label_indices_test)
        dist_test_all = []
        dice_test_all = []
        for k in range(num_miniBatch_test):
            idx_test = [testIndices[i] for i in range(miniBatchSize*k, miniBatchSize*(k+1))]
            idx_label_test = [label_indices_test[i] for i in range(miniBatchSize*k, miniBatchSize*(k+1))]
            testFeed = {movingImage_ph: feeder_moving.get_image_batch(idx_test),
                        targetImage_ph: feeder_target.get_image_batch(idx_test),
                        movingLabel_ph: feeder_moving.get_binary_batch(idx_test, idx_label_test),
                        targetLabel_ph: feeder_target.get_binary_batch(idx_test, idx_label_test),
                        movingTransform_ph: transform_identity,
                        targetTransform_ph: transform_identity}

            if flag_debugMore:
                grid_sample_global_test, displacement_local_test = sess.run([grid_sample_global, displacement_local], feed_dict=testFeed)
                fid_debug_data.create_dataset('/displacement_local_test_k%d/' % k, displacement_local_test.shape, dtype=displacement_local_test.dtype, data=displacement_local_test)
                if (network_type == 'pre-warp') | (network_type == 'two-stream'):
                    fid_debug_data.create_dataset('/grid_sample_global_test_k%d/' % k, grid_sample_global_test.shape, dtype=grid_sample_global_test.dtype, data=grid_sample_global_test)

            dice_test, dist_test, movingVol_test, targetVol_test = sess.run([dice, dist, movingVol, targetVol], feed_dict=testFeed)
            # fid_debug_data.create_dataset('/dice_test_k%d/' % k, data=dice_test)
            # fid_debug_data.create_dataset('/dist_test_k%d/' % k, data=dist_test)
            if flag_debugPrint:
                print('DEBUG-PRINT: idx_test: %s' % idx_test)
                print('DEBUG-PRINT: idx_test: %s' % idx_test, flush=True, file=fid_output_info)
                print('DEBUG-PRINT: idx_label_test: %s' % idx_label_test)
                print('DEBUG-PRINT: idx_label_test: %s' % idx_label_test, flush=True, file=fid_output_info)
                print('***test*** Dice: %s' % dice_test)
                print('***test*** Dice: %s' % dice_test, flush=True, file=fid_output_info)
                print('***test*** Distance: %s' % dist_test)
                print('***test*** Distance: %s' % dist_test, flush=True, file=fid_output_info)
                print('***test*** movingVol: %s' % movingVol_test)
                print('***test*** movingVol: %s' % movingVol_test, flush=True, file=fid_output_info)
                print('***test*** targetVol: %s' % targetVol_test)
                print('***test*** targetVol: %s' % targetVol_test, flush=True, file=fid_output_info)
            # collect for summary
            dist_test_all += dist_test.tolist()
            dice_test_all += dice_test[[idx_label_test[i] == 0 for i in range(miniBatchSize)]].tolist()

        # summary statistics
        dist_test_all = dist_test_all[0:num_labels_test]
        dice_test_all = dice_test_all[0:num_cases_test]
        tre_test = [np.sqrt(np.nanmean([dist_test_all[i]**2 if testIndices[i]==j else np.nan for i in range(num_labels_test)])) for j in testCaseIndices]
        fid_debug_data.create_dataset('/dice_test_all/', data=dice_test_all)
        fid_debug_data.create_dataset('/dist_test_all/', data=dist_test_all)
        fid_debug_data.create_dataset('/tre_test/', data=tre_test)
        statisticsDice = [np.nanmean(dice_test_all), np.nanstd(dice_test_all), np.nanmedian(dice_test_all),
                          np.nanpercentile(dice_test_all,5), np.nanpercentile(dice_test_all,10),
                          np.nanpercentile(dice_test_all,90), np.nanpercentile(dice_test_all,95)]
        statisticsDist = [np.nanmean(dist_test_all), np.nanstd(dist_test_all), np.nanmedian(dist_test_all),
                          np.nanpercentile(dist_test_all, 5), np.nanpercentile(dist_test_all, 10),
                          np.nanpercentile(dist_test_all, 90), np.nanpercentile(dist_test_all, 95)]
        statisticsTREs = [np.nanmean(tre_test), np.nanstd(tre_test), np.nanmedian(tre_test),
                          np.nanpercentile(tre_test, 5), np.nanpercentile(tre_test, 10),
                          np.nanpercentile(tre_test, 90), np.nanpercentile(tre_test, 95)]
        print('***test-summary *** Dice stats: %s' % statisticsDice)
        print('***test-summary *** Dice stats: %s' % statisticsDice, flush=True, file=fid_output_info)
        print('***test-summary *** Distance stats: %s' % statisticsDist)
        print('***test-summary *** Distance stats: %s' % statisticsDist, flush=True, file=fid_output_info)
        print('***test-summary *** TRE stats: %s' % statisticsTREs)
        print('***test-summary *** TRE stats: %s' % statisticsTREs, flush=True, file=fid_output_info)
        # ----test----

        # flush in the end
        fid_debug_data.flush()
        fid_debug_data.close()
        print('Debug data saved at Step %d' % step)
        print('Debug data saved at Step %d' % step, flush=True, file=fid_output_info)
        # NB. do not use continue here!
# ---------- End of Computational Graph ----------

fid_output_info.close()
