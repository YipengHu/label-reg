import tensorflow as tf
import config

import helper
import util
import network as net
import numpy as np


data_resampler = helper.DataResampler

reg_net = net.DDFNet




if not (totalDataSize == totalDataSize2):
    raise Exception('moving and target data should have the same size!')
feeder_target = tfhelpers.DataFeeder(h5fn_image_target, h5fn_label_target)
feeder_moving = tfhelpers.DataFeeder(h5fn_image_moving, h5fn_label_moving)
size_1 = tfhelpers.get_padded_shape(size_target, 2)
size_2 = tfhelpers.get_padded_shape(size_target, 4)
size_3 = tfhelpers.get_padded_shape(size_target, 8)
size_4 = tfhelpers.get_padded_shape(size_target, 16)
vol_target = size_target[0] * size_target[1] * size_target[2]

testCaseIndices, testIndices, label_indices_test, num_cases_test, num_labels_test, num_miniBatch_test, trainIndices, num_miniBatch \
    = tfhelpers.setup_cross_validation_simple(totalDataSize, num_fold, idx_fold, miniBatchSize, num_important)

# pre-computing for graph
transform_identity = tfhelpers.initial_transform_generator(miniBatchSize)
grid_reference = tftools.get_reference_grid(size_target)
grid_moving = tf.stack([tftools.get_reference_grid(size_moving)]*miniBatchSize, axis=0)  # tf.expand_dims(tftools.get_reference_grid(size_moving), axis=0)




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

