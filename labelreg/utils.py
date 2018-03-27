import tensorflow as tf


def warp_image_affine(vol, theta):
    return resample_linear(vol, warp_grid(get_reference_grid(vol.get_shape()[1:4]), theta))


def warp_grid(grid, theta):
    # grid=grid_reference
    num_batch = int(theta.get_shape()[0])
    theta = tf.cast(tf.reshape(theta, (-1, 3, 4)), 'float32')
    size_i = int(grid.get_shape()[0])
    size_j = int(grid.get_shape()[1])
    size_k = int(grid.get_shape()[2])
    grid = tf.concat([tf.transpose(tf.reshape(grid, [-1, 3])), tf.ones([1, size_i*size_j*size_k])], axis=0)
    grid = tf.reshape(tf.tile(tf.reshape(grid, [-1]), [num_batch]), [num_batch, 4, -1])
    grid_warped = tf.matmul(theta, grid)
    return tf.reshape(tf.transpose(grid_warped, [0, 2, 1]), [num_batch, size_i, size_j, size_k, 3])


def resample_linear(inputs, sample_coords):

    input_size = inputs.get_shape().as_list()[1:-1]
    spatial_rank = inputs.get_shape().ndims - 2
    xy = tf.unstack(sample_coords, axis=len(sample_coords.get_shape())-1)
    index_voxel_coords = [tf.floor(x) for x in xy]

    def boundary_replicate(sample_coords0, input_size0):
        return tf.maximum(tf.minimum(sample_coords0, input_size0 - 1), 0)
    spatial_coords = [boundary_replicate(tf.cast(x, tf.int32), input_size[idx]) for idx, x in enumerate(index_voxel_coords)]
    spatial_coords_plus1 = [boundary_replicate(tf.cast(x+1., tf.int32), input_size[idx]) for idx, x in enumerate(index_voxel_coords)]

    weight = [tf.expand_dims(x - tf.cast(i, tf.float32), -1) for x, i in zip(xy, spatial_coords)]
    weight_c = [tf.expand_dims(tf.cast(i, tf.float32) - x, -1) for x, i in zip(xy, spatial_coords_plus1)]

    sz = spatial_coords[0].get_shape().as_list()
    batch_coords = tf.tile(tf.reshape(tf.range(sz[0]), [sz[0]] + [1] * (len(sz) - 1)), [1] + sz[1:])
    sc = (spatial_coords, spatial_coords_plus1)
    binary_codes = [[int(c) for c in format(i, '0%ib' % spatial_rank)] for i in range(2**spatial_rank)]

    make_sample = lambda bc: tf.gather_nd(inputs, tf.stack([batch_coords] + [sc[c][i] for i, c in enumerate(bc)], -1))
    samples = [make_sample(bc) for bc in binary_codes]

    def pyramid_combination(samples0, weight0, weight_c0):
        if len(weight0) == 1:
            return samples0[0]*weight_c0[0]+samples0[1]*weight0[0]
        else:
            return pyramid_combination(samples0[::2], weight0[:-1], weight_c0[:-1]) * weight_c0[-1] + \
                   pyramid_combination(samples0[1::2], weight0[:-1], weight_c0[:-1]) * weight0[-1]

    return pyramid_combination(samples, weight, weight_c)


def get_reference_grid(grid_size):
    return tf.to_float(tf.stack(tf.meshgrid(
        [i for i in range(grid_size[0])],
        [j for j in range(grid_size[1])],
        [k for k in range(grid_size[2])],
        indexing='ij'), axis=3))


def gradient_dx(fv):
    return (fv[:, 2:, 1:-1, 1:-1] - fv[:, :-2, 1:-1, 1:-1]) / 2


def gradient_dy(fv):
    return (fv[:, 1:-1, 2:, 1:-1] - fv[:, 1:-1, :-2, 1:-1]) / 2


def gradient_dz(fv):
    return (fv[:, 1:-1, 1:-1, 2:] - fv[:, 1:-1, 1:-1, :-2]) / 2


def gradient_txyz(Txyz, fn):
    return tf.stack([fn(Txyz[:, :, :, :, i]) for i in [0, 1, 2]], axis=4)


def compute_gradient_norm(displacement, flag_l1=False):
    dTdx = gradient_txyz(displacement, gradient_dx)
    dTdy = gradient_txyz(displacement, gradient_dy)
    dTdz = gradient_txyz(displacement, gradient_dz)
    if flag_l1:
        norms = tf.abs(dTdx) + tf.abs(dTdy) + tf.abs(dTdz)
    else:
        norms = dTdx**2 + dTdy**2 + dTdz**2
    return tf.reduce_mean(norms, [1, 2, 3, 4])


def compute_bending_energy(displacement):
    dTdx = gradient_txyz(displacement, gradient_dx)
    dTdy = gradient_txyz(displacement, gradient_dy)
    dTdz = gradient_txyz(displacement, gradient_dz)
    dTdxx = gradient_txyz(dTdx, gradient_dx)
    dTdyy = gradient_txyz(dTdy, gradient_dy)
    dTdzz = gradient_txyz(dTdz, gradient_dz)
    dTdxy = gradient_txyz(dTdx, gradient_dy)
    dTdyz = gradient_txyz(dTdy, gradient_dz)
    dTdxz = gradient_txyz(dTdx, gradient_dz)
    return tf.reduce_mean(dTdxx**2 + dTdyy**2 + dTdzz**2 + 2*dTdxy**2 + 2*dTdxz**2 + 2*dTdyz**2, [1, 2, 3, 4])


def compute_binary_dice(input1, input2):
    mask1 = input1 >= 0.5
    mask2 = input2 >= 0.5
    vol1 = tf.reduce_sum(tf.to_float(mask1), axis=[1, 2, 3, 4])
    vol2 = tf.reduce_sum(tf.to_float(mask2), axis=[1, 2, 3, 4])
    dice = tf.reduce_sum(tf.to_float(mask1 & mask2), axis=[1, 2, 3, 4])*2 / (vol1+vol2)
    return dice, vol1, vol2


def compute_centroid_distance(input1, input2, grid):
    def compute_centroid(mask, grid0):
        return tf.stack([tf.reduce_mean(tf.boolean_mask(grid0, mask[i,:,:,:,0]>=0.5), axis=0) for i in range(mask.shape[0].value)], axis=0)
    c1 = compute_centroid(input1, grid)
    c2 = compute_centroid(input2, grid)
    return tf.sqrt(tf.reduce_sum(tf.square(c1-c2), axis=1))


def weighted_binary_cross_entropy(ts, ps, pw=1, eps=1e-6):
    ps = tf.clip_by_value(ps, eps, 1-eps)
    return -tf.reduce_sum(tf.concat([ts*pw, 1-ts], axis=4)*tf.log(tf.concat([ps, 1-ps], axis=4)), axis=4, keep_dims=True)


def dice_simple(ts, ps, eps_vol=1e-6):
    numerator = tf.reduce_sum(ts*ps, axis=[1, 2, 3, 4]) * 2
    denominator = tf.reduce_sum(ts, axis=[1, 2, 3, 4]) + tf.reduce_sum(ps, axis=[1, 2, 3, 4])+eps_vol
    return numerator/denominator


def dice_generalised(ts, ps, weights):
    ts2 = tf.concat([ts, 1-ts], axis=4)
    ps2 = tf.concat([ps, 1-ps], axis=4)
    numerator = 2 * tf.reduce_sum(tf.reduce_sum(ts2*ps2, axis=[1, 2, 3]) * weights, axis=1)
    denominator = tf.reduce_sum((tf.reduce_sum(ts2, axis=[1, 2, 3]) + tf.reduce_sum(ps2, axis=[1, 2, 3])) * weights, axis=1)
    return numerator/denominator


def jaccard_simple(ts, ps, eps_vol=1e-6):
    numerator = tf.reduce_sum(ts*ps, axis=[1, 2, 3, 4])
    denominator = tf.reduce_sum(tf.square(ts), axis=[1, 2, 3, 4]) + tf.reduce_sum(tf.square(ps), axis=[1, 2, 3, 4]) - numerator + eps_vol
    return numerator/denominator


def gauss_kernel1d(sigma):
    if sigma == 0:
        return 0
    else:
        tail = int(sigma*3)
        k = tf.exp([-0.5*x**2/sigma**2 for x in range(-tail, tail+1)])
        return k / tf.reduce_sum(k)


def cauchy_kernel1d(sigma):  # this is an approximation
    if sigma == 0:
        return 0
    else:
        tail = int(sigma*5)
        # k = tf.reciprocal(([((x/sigma)**2+1)*sigma*3.141592653589793 for x in range(-tail, tail+1)]))
        k = tf.reciprocal([((x/sigma)**2+1) for x in range(-tail, tail + 1)])
        return k / tf.reduce_sum(k)


def separable_filter3d(vol, kernel):
    if kernel == 0:
        return vol
    else:
        strides = [1, 1, 1, 1, 1]
        return tf.nn.conv3d(tf.nn.conv3d(tf.nn.conv3d(
            vol,
            tf.reshape(kernel, [-1,1,1,1,1]), strides, "SAME"),
            tf.reshape(kernel, [1,-1,1,1,1]), strides, "SAME"),
            tf.reshape(kernel, [1,1,-1,1,1]), strides, "SAME")


def separable_filternd(vol, kernel):
    if kernel == 0:
        return vol
    else:
        strides = [1, 1, 1, 1, 1]
        return tf.nn.conv3d(tf.nn.conv3d(tf.nn.conv3d(
            vol,
            tf.reshape(kernel, [-1,1,1,1,1]), strides, "SAME"),
            tf.reshape(kernel, [1,-1,1,1,1]), strides, "SAME"),
            tf.reshape(kernel, [1,1,-1,1,1]), strides, "SAME")


def mixture_filter3d(vol, sigmas, kernel_type):
    if kernel_type == 'cauchy':
        return tf.reduce_mean(tf.concat([separable_filter3d(vol, cauchy_kernel1d(s)) for s in sigmas], axis=4), axis=4, keep_dims=True)
    elif kernel_type == 'gauss':
        return tf.reduce_mean(tf.concat([separable_filter3d(vol, gauss_kernel1d(s)) for s in sigmas], axis=4), axis=4, keep_dims=True)


def single_scale_loss(targetLabel, movingLabel_warped, loss_type):
    if loss_type == 'cross-entropy':
        label_loss_batch = tf.reduce_mean(weighted_binary_cross_entropy(targetLabel, movingLabel_warped), axis=[1, 2, 3, 4])
    elif loss_type == 'mean-squared':
        label_loss_batch = tf.reduce_mean(tf.squared_difference(targetLabel, movingLabel_warped), axis=[1, 2, 3, 4])
    elif loss_type == 'dice':
        label_loss_batch = 1 - dice_simple(targetLabel, movingLabel_warped)
    elif loss_type == 'jaccard':
        label_loss_batch = 1 - jaccard_simple(targetLabel, movingLabel_warped)
    else:
        raise Exception('Not recognised label correspondence loss!')
    return label_loss_batch


def multi_scale_loss(targetLabel, movingLabel_warped, loss_type, loss_scales):
    label_loss_all = tf.stack(
        [single_scale_loss(
            separable_filter3d(targetLabel, gauss_kernel1d(s)),
            separable_filter3d(movingLabel_warped, gauss_kernel1d(s)), loss_type)
            for s in loss_scales],
        axis=1)
    return tf.reduce_mean(label_loss_all, axis=1)

