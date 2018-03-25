
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
