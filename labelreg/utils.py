import tensorflow as tf
import spatial_transformer_3d


class AggregateBatchMeanVariance:
    def __init__(self, values):
        self.mean = tf.reduce_mean(values)
        self.count = tf.size(values)
        self.m2 = tf.reduce_sum(tf.square(values - self.mean))
        self.var = self.m2 / (self.count-1)

    def update(self, new_values):
        new_mean = tf.reduce_mean(new_values)
        new_count = tf.size(new_values)
        new_m2 = tf.reduce_sum(tf.square(new_values - new_mean))
        # new_var = new_m2 / (new_count-1)
        delta = new_mean - self.mean
        self.count = self.count+new_count
        delta_mean = delta*new_count/self.count
        self.mean = self.mean + delta_mean
        self.m2 = self.m2 + new_m2 + delta*self.count*delta_mean
        self.var = self.m2 / (self.count-1)


def random_transform2(vol1, vol2, transform_vector, output_size):
    vol1 = spatial_transformer_3d.transformer(vol1, transform_vector, output_size, 'linear')
    vol2 = tf.to_int32(spatial_transformer_3d.transformer(vol2, transform_vector, output_size, 'nearest'))
    return vol1, vol2


def random_transform1(vol1, transform_vector, output_size):
    vol1 = spatial_transformer_3d.transformer(vol1, transform_vector, output_size, 'linear')
    return vol1


def random_transform3(vol1, vol2, vol3, transform_vector, output_size):
    vol1 = spatial_transformer_3d.transformer(vol1, transform_vector, output_size, 'linear')
    vol2 = tf.to_int32(spatial_transformer_3d.transformer(vol2, transform_vector, output_size, 'nearest'))
    vol3 = spatial_transformer_3d.transformer(vol3, transform_vector, output_size, 'linear')
    return vol1, vol2, vol3


def random_transform_ddf(ddf, transform_vector):
    ddf_size = ddf.get_shape().as_list()
    grid0 = tf.tile(tf.expand_dims(get_reference_grid(ddf_size[1:4]),axis=0),[ddf_size[0],1,1,1,1])
    grid1 = spatial_transformer_3d.transformer(grid0 + ddf, transform_vector, ddf_size[1:4], 'linear')
    return grid1 - grid0


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





def resample_linear(inputs, sample_coords, boundary='ZERO'):

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


def get_center_grid(grid_size):
    return tf.to_float(tf.stack(tf.meshgrid(
        [i-grid_size[0]/2 for i in range(grid_size[0])],
        [j-grid_size[1]/2 for j in range(grid_size[1])],
        [k-grid_size[2]/2 for k in range(grid_size[2])],
        indexing='ij'), axis=3))


def displacement_filtering(displacement, kernel, size_out):
    # todo: add conv3d_transpose for ffd approximation
    return tf.concat([tf.nn.conv3d(tf.expand_dims(displacement[:,:,:,:,0], axis=4), kernel, [1, 1, 1, 1, 1], "SAME"),
                      tf.nn.conv3d(tf.expand_dims(displacement[:,:,:,:,1], axis=4), kernel, [1, 1, 1, 1, 1], "SAME"),
                      tf.nn.conv3d(tf.expand_dims(displacement[:,:,:,:,2], axis=4), kernel, [1, 1, 1, 1, 1], "SAME")],
                     axis=4)


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


def compute_gradient_norm_local(displacement, flag_l1=False):
    dTdx = gradient_txyz(displacement, gradient_dx)
    dTdy = gradient_txyz(displacement, gradient_dy)
    dTdz = gradient_txyz(displacement, gradient_dz)
    if flag_l1:
        norms = tf.abs(dTdx) + tf.abs(dTdy) + tf.abs(dTdz)
    else:
        norms = dTdx**2 + dTdy**2 + dTdz**2
    return norms


def compute_gradient_tensor(displacement, flag_l1=False):
    dTdx = gradient_txyz(displacement, gradient_dx)
    dTdy = gradient_txyz(displacement, gradient_dy)
    dTdz = gradient_txyz(displacement, gradient_dz)
    return tf.concat([dTdx,dTdy,dTdz], axis=4)


def compute_small_strain_tensor(ddf, quadratic=True):
    # compute the deformation gradient tensor in inner matrix
    dTdx = gradient_txyz(ddf, gradient_dx)
    dTdy = gradient_txyz(ddf, gradient_dy)
    dTdz = gradient_txyz(ddf, gradient_dz)
    # dD = tf.concat([dTdx,dTdy,dTdz], axis=5)  # F - I

    if quadratic:
        epsilon = tf.stack([
            dTdx[:, :, :, :, 0] + 0.5 * tf.reduce_sum(tf.square(dTdx), axis=4),
            dTdy[:, :, :, :, 1] + 0.5 * tf.reduce_sum(tf.square(dTdy), axis=4),
            dTdz[:, :, :, :, 2] + 0.5 * tf.reduce_sum(tf.square(dTdz), axis=4),
            0.5 * (dTdx[:, :, :, :, 1] + dTdy[:, :, :, :, 0] +
                   dTdx[:, :, :, :, 0]*dTdy[:, :, :, :, 0] +
                   dTdx[:, :, :, :, 1]*dTdy[:, :, :, :, 1] +
                   dTdx[:, :, :, :, 2]*dTdy[:, :, :, :, 2]),
            0.5 * (dTdx[:, :, :, :, 2] + dTdz[:, :, :, :, 0] +
                   dTdx[:, :, :, :, 0]*dTdz[:, :, :, :, 0] +
                   dTdx[:, :, :, :, 1]*dTdz[:, :, :, :, 1] +
                   dTdx[:, :, :, :, 2]*dTdz[:, :, :, :, 2]),
            0.5 * (dTdz[:, :, :, :, 1] + dTdy[:, :, :, :, 2] +
                   dTdz[:, :, :, :, 0]*dTdy[:, :, :, :, 0] +
                   dTdz[:, :, :, :, 1]*dTdy[:, :, :, :, 1] +
                   dTdz[:, :, :, :, 2]*dTdy[:, :, :, :, 2])], axis=4)

    else:
        epsilon = tf.stack([dTdx[:, :, :, :, 0],
                            dTdy[:, :, :, :, 1],
                            dTdz[:, :, :, :, 2],
                            (dTdx[:, :, :, :, 1] + dTdy[:, :, :, :, 0]) * 0.5,
                            (dTdx[:, :, :, :, 2] + dTdz[:, :, :, :, 0]) * 0.5,
                            (dTdy[:, :, :, :, 2] + dTdz[:, :, :, :, 1]) * 0.5], axis=4)

    return epsilon


def compute_finite_strain_tensor(ddf, method='eigen3'):
    # compute the deformation gradient tensor in inner matrix
    dTdx = tf.expand_dims(gradient_txyz(ddf, gradient_dx), axis=5)
    dTdy = tf.expand_dims(gradient_txyz(ddf, gradient_dy), axis=5)
    dTdz = tf.expand_dims(gradient_txyz(ddf, gradient_dz), axis=5)
    # plus identity
    F = tf.concat([dTdx,dTdy,dTdz], axis=5) + tf.eye(3)

    if method == 'eigen3':   # eigv, Q = tf.self_adjoint_eig(tf.transpose(F,[0,1,2,3,5,4]) * F):
        eigv, Q = symmetric_eig_3by3(tf.transpose(F,[0,1,2,3,5,4]) * F)
        # now the stretch tensor
        Q2 = tf.transpose(Q,[0,1,2,3,5,4]) * Q
        sqrt_e = tf.sqrt(eigv)

    elif method == 'eigen':   # eigv, Q = tf.self_adjoint_eig(tf.transpose(F,[0,1,2,3,5,4]) * F):
        # eigen-decomposition: tensor[...,:,:] * v[..., :,i] = e[..., i] * v[...,:,i]
        eigv, Q = tf.self_adjoint_eig(tf.transpose(F,[0,1,2,3,5,4]) * F)
        # now the stretch tensor
        Q2 = tf.transpose(Q,[0,1,2,3,5,4]) * Q
        sqrt_e = tf.sqrt(eigv)

    elif method == 'svd':
        sqrt_e, _, v = tf.svd(F)
        Q2 = v * tf.transpose(v,[0,1,2,3,5,4])  # left: Q2 = w * tf.transpose(w,[0,1,2,3,5,4])

    else:
        raise Exception('Unknown polar decomposition methods!')

    epsilon = tf.stack([sqrt_e[:, :, :, :, 0] * Q2[:, :, :, :, 0, 0] - 1,
                        sqrt_e[:, :, :, :, 1] * Q2[:, :, :, :, 1, 1] - 1,
                        sqrt_e[:, :, :, :, 2] * Q2[:, :, :, :, 2, 2] - 1,
                        sqrt_e[:, :, :, :, 1] * Q2[:, :, :, :, 1, 0],
                        sqrt_e[:, :, :, :, 2] * Q2[:, :, :, :, 2, 0],
                        sqrt_e[:, :, :, :, 2] * Q2[:, :, :, :, 2, 1]], axis=4)
    return epsilon


def det_3by3(x):
    return x[:,:,:,:,0,0]*(x[:,:,:,:,1,1]*x[:,:,:,:,2,2]-x[:,:,:,:,1,2]*x[:,:,:,:,2,1]) \
           - x[:,:,:,:,0,1]*(x[:,:,:,:,1,0]*x[:,:,:,:,2,2]-x[:,:,:,:,1,2]*x[:,:,:,:,2,0]) \
           + x[:,:,:,:,0,2]*(x[:,:,:,:,1,0]*x[:,:,:,:,2,1]-x[:,:,:,:,1,1]*x[:,:,:,:,2,0])


def compute_normal_eigenvector(x,e):
    # since the x must be normal
    sz = x.get_shape().as_list()
    u = x - tf.matrix_diag(tf.tile(tf.expand_dims(e, axis=4), [1, 1, 1, 1, 3]))
    cross_rows = tf.stack([
        tf.cross(u[:, :, :, :, 0, :], u[:, :, :, :, 1, :]),
        tf.cross(u[:, :, :, :, 0, :], u[:, :, :, :, 2, :]),
        tf.cross(u[:, :, :, :, 1, :], u[:, :, :, :, 2, :])], axis=5)
    ds = tf.reduce_sum(tf.square(cross_rows), axis=4)
    dmax = tf.reduce_max(ds, axis=4, keep_dims=True)
    imax = tf.stack([tf.tile(tf.reshape(tf.argmax(ds, axis=4), shape=[-1,]), [3,]),
                     tf.range(sz[0]*sz[1]*sz[2]*sz[3]*sz[4], dtype='int64')], axis=1)
    v = tf.reshape(tf.gather_nd(tf.reshape(cross_rows,[-1,3]),imax),sz[0:5]) / tf.sqrt(dmax)
    return v


def symmetric_eig_3by3(x):
    # tridiagonal solution
    p1 = tf.square(x[:,:,:,:,0,1]) + tf.square(x[:,:,:,:,0,2]) + tf.square(x[:,:,:,:,1,2])
    q = tf.trace(x)/3
    p2 = tf.reduce_sum(tf.square(tf.matrix_diag_part(x)-tf.expand_dims(q,axis=4)),axis=4) + 2*p1
    p = tf.sqrt(p2/6)
    B = tf.expand_dims(tf.expand_dims(tf.reciprocal(p),axis=4),axis=5) * (x-tf.matrix_diag(tf.tile(tf.expand_dims(q,axis=4),[1,1,1,1,3])))
    r = det_3by3(B) / 2
    r = tf.clip_by_value(r, -1, 1)
    phi = tf.acos(r) / 3
    e1 = q + 2*p*tf.cos(phi)
    e3 = q + 2*p*tf.cos(phi+2.0943951023931953)  # 2*3.141592653589793/3
    e2 = 3*q - e1 - e3
    v1 = compute_normal_eigenvector(x, e1)
    v2 = compute_normal_eigenvector(x, e2)
    v3 = compute_normal_eigenvector(x, e3)

    return tf.stack([e1,e2,e3], axis=4), tf.stack([v1,v2,v3], axis=5)


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


def compute_ddf_features(ddf, feature_type, feature_size):
    if feature_type == 'strain-tensor-quadratic':
        ddf_feature = compute_small_strain_tensor(ddf, quadratic=True)
    elif feature_type == 'strain-tensor':
        ddf_feature = compute_small_strain_tensor(ddf, quadratic=False)
    elif feature_type == 'strain-tensor-finite':
        ddf_feature = compute_finite_strain_tensor(ddf)
    elif feature_type == 'gradient-tensor':
        ddf_feature = compute_gradient_tensor(ddf)
    elif feature_type == 'displacement':
        ddf_feature = ddf
    elif feature_type == 'gradient-l2':
        ddf_feature = compute_gradient_norm_local(ddf, flag_l1=False)
    elif feature_type == 'gradient-l1':
        ddf_feature = compute_gradient_norm_local(ddf, flag_l1=True)
    else:
        raise Exception('unknown ddf feature type!')

    ddf_size = ddf.get_shape().as_list()[1:4]
    if not (ddf_size == feature_size):
        # sigma = sum([feature_size[i]/ddf_size[i]*0.5 for i in [0,1,2]])/3   # empirical anti-aliasing
        # ddf_features = seperable_filter3d(ddf_feature, gauss_kernel1d(sigma))
        ddf_feature = resize_volume(ddf_feature, feature_size)

    return ddf_feature


def hierarchy_regulariser(displacement, weights, size_hierarchy, fn):
    energy = tf.constant(0.0)
    for idx in range(len(weights)):
        if weights[idx] > 0:
            if idx == 0:
                energy_batches = fn(displacement)
            else:
                energy_batches = fn(resize_volume(displacement, size_hierarchy[idx]))
            energy += tf.reduce_mean(energy_batches) * weights[idx]
    return energy


def compute_dice(input1, input2):
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


def compute_positive_weight(p_pos, p_neg, alpha):
    if alpha == 0:
        return 1.0

    magn = p_pos+p_neg
    s_pos = 1-alpha+0.5*alpha/(p_pos/magn)
    s_neg = 1-alpha+0.5*alpha/(p_neg/magn)
    return s_pos/s_neg


def weighted_binary_cross_entropy(ts, ps, pw=1, eps=1e-6):
    ps = tf.clip_by_value(ps, eps, 1-eps)
    return -tf.reduce_sum(tf.concat([ts*pw, 1-ts], axis=4)*tf.log(tf.concat([ps, 1-ps], axis=4)), axis=4, keep_dims=True)


def dice_simple(ts, ps, eps_vol=1e-6):
    numerator = tf.reduce_sum(ts*ps, axis=[1, 2, 3, 4]) * 2
    denominator = tf.reduce_sum(ts, axis=[1, 2, 3, 4]) + tf.reduce_sum(ps, axis=[1, 2, 3, 4])+eps_vol
    return numerator/denominator


def dice_generalised(ts, ps, weights):
    # volumes = tf.reduce_sum(ts, axis=[1, 2, 3, 4])  # only using ground-truth volumes as weights
    # weights = tf.stack([
    #     tf.reciprocal(tf.square(volumes)),
    #     tf.reciprocal(tf.square(tf.to_float(tf.reduce_prod(tf.shape(ts)[1:4])) - volumes))], axis=1)
    ts2 = tf.concat([ts, 1-ts], axis=4)
    ps2 = tf.concat([ps, 1-ps], axis=4)
    numerator = 2 * tf.reduce_sum(tf.reduce_sum(ts2*ps2, axis=[1, 2, 3]) * weights, axis=1)
    denominator = tf.reduce_sum((tf.reduce_sum(ts2, axis=[1, 2, 3]) + tf.reduce_sum(ps2, axis=[1, 2, 3])) * weights, axis=1)
    return numerator/denominator


def jaccard_simple(ts, ps, eps_vol=1e-6):
    numerator = tf.reduce_sum(ts*ps, axis=[1, 2, 3, 4])
    denominator = tf.reduce_sum(tf.square(ts), axis=[1, 2, 3, 4]) + tf.reduce_sum(tf.square(ps), axis=[1, 2, 3, 4]) - numerator + eps_vol
    return numerator/denominator


def dice_distance(ts, ps, dw, grid, eps_overlap=1e-6):
    dice = dice_simple(ts, ps)
    m1 = tf.reduce_sum(ts, axis=[1,2,3,4])+1e-6
    m2 = tf.reduce_sum(ps, axis=[1,2,3,4])+1e-6
    c1 = tf.reduce_sum(ts * tf.expand_dims(grid, axis=0), axis=[1,2,3,4]) / m1
    c2 = tf.reduce_sum(ps * tf.expand_dims(grid, axis=0), axis=[1,2,3,4]) / m2
    ds = tf.sqrt(tf.reduce_sum(tf.square(c1-c2), axis=1, keep_dims=True)) * dw
    return dice-tf.floor(1-dice)*ds, dice, ds



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


def seperable_filter3d(vol, kernel):
    if kernel == 0:
        return vol
    else:
        strides = [1, 1, 1, 1, 1]
        return tf.nn.conv3d(tf.nn.conv3d(tf.nn.conv3d(
            vol,
            tf.reshape(kernel, [-1,1,1,1,1]), strides, "SAME"),
            tf.reshape(kernel, [1,-1,1,1,1]), strides, "SAME"),
            tf.reshape(kernel, [1,1,-1,1,1]), strides, "SAME")


def seperable_filternd(vol, kernel):
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
        return tf.reduce_mean(tf.concat([seperable_filter3d(vol, cauchy_kernel1d(s)) for s in sigmas], axis=4), axis=4, keep_dims=True)
    elif kernel_type == 'gauss':
        return tf.reduce_mean(tf.concat([seperable_filter3d(vol, gauss_kernel1d(s)) for s in sigmas], axis=4), axis=4, keep_dims=True)


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
            seperable_filter3d(targetLabel, gauss_kernel1d(s)),
            seperable_filter3d(movingLabel_warped, gauss_kernel1d(s)), loss_type)
            for s in loss_scales],
        axis=1)
    return tf.reduce_mean(label_loss_all, axis=1)


# for adversarial learning
def leaky_relu(x, alpha=0.2, name="leaky_relu"):
    with tf.variable_scope(name):
        # return tf.maximum(tf.minimum(0.0, alpha * x), x)
        return 0.5 * (1 + alpha) * x + 0.5 * (1 - alpha) * abs(x)


def exclude_post_affine_backward(input_ddf):
    sz = input_ddf.get_shape().as_list()
    nn = sz[1]*sz[2]*sz[3]
    grid1 = tf.reshape(get_reference_grid(sz[1:4]), [-1,3])  # deformed
    grid0 = grid1 + tf.reshape(input_ddf,[sz[0],-1,3])  # undeformed
    affine = [tf.matrix_solve_ls(tf.concat([grid1, tf.ones([nn,1])], axis=1),
                                 tf.concat([grid0[i,], tf.ones([nn, 1])], axis=1)) for i in range(sz[0])]
    grid_affine = tf.stack(
        [tf.matmul(tf.concat([grid1, tf.ones([nn,1])], axis=1), affine[i])[:,0:3] for i in range(sz[0])], axis=0)

    return tf.reshape(grid0 - grid_affine, sz)  # return inverse ddf


def exclude_pre_affine_backward(input_ddf):

    sz = input_ddf.get_shape().as_list()
    nn = sz[1]*sz[2]*sz[3]
    grid1 = get_reference_grid(sz[1:4])  # deformed
    grid0 = tf.reshape(grid1 + input_ddf,[sz[0],-1,3])  # undeformed
    affine = [tf.matrix_solve_ls(tf.concat([grid0[i,], tf.ones([nn,1])], axis=1),
                                 tf.concat([tf.reshape(grid1,[-1,3]), tf.ones([nn,1])], axis=1)) for i in range(sz[0])]
    grid_affine = tf.stack(
        [tf.matmul(tf.concat([grid0[i,], tf.ones([nn,1])], axis=1), affine[i])[:,0:3] for i in range(sz[0])], axis=0)

    return tf.reshape(grid_affine,sz) - grid1  # return inverse ddf


def compute_bounding_box(mask,ddf=0):
    mbsz = mask.shape[0].value
    grid = get_reference_grid([mask.shape[i].value for i in [1, 2, 3]])
    if ddf == 0:
        pts = [tf.boolean_mask(grid, mask[i,:,:,:,0]>=0.5) for i in range(mbsz)]
    else:
        grid += ddf
        pts = [tf.boolean_mask(grid[i,:,:,:], mask[i,:,:,:,0] >= 0.5) for i in range(mbsz)]

    # centroids = tf.stack([tf.reduce_mean(pts[i], axis=0) for i in range(sz[0])], axis=0)
    vmax = tf.stack([tf.reduce_max(pts[i], axis=0) for i in range(mbsz)], axis=0)
    vmin = tf.stack([tf.reduce_min(pts[i], axis=0) for i in range(mbsz)], axis=0)
    length = tf.reshape(vmax-vmin, [mbsz,1,1,1,3])
    centre = length/2 + tf.reshape(vmin, [mbsz,1,1,1,3])
    return length, centre, grid


def compute_st(c0,c1,l0,l1):
    s = l1/l0
    t = c1 - c0*s
    return s, t


def combine_st(s0,t0,s1,t1):
    s = s0*s1
    t = t0*s1 + t1
    return s, t


def normalise_ddf_sample_moving(mask_target, mask_motion, input_ddf, scale_ratio=1, local_only=True):

    # sample first
    len_target1, cen_target1, grid_target1 = compute_bounding_box(mask_target)
    len_motion1, cen_motion1, _ = compute_bounding_box(mask_motion)  # deformed
    # deformation
    s_t1, t_t1 = compute_st(cen_target1, cen_motion1, len_target1, len_motion1)
    grid_sample = grid_target1 * s_t1 + t_t1
    resampled_ddf = resample_linear(input_ddf, grid_sample) / scale_ratio

    if local_only:  # extract affine
        resampled_ddf = exclude_post_affine_backward(resampled_ddf)

    return resampled_ddf


def normalise_ddf_exclusive(mask_moving, mask_target, mask_motion, input_ddf, scale_ratio=1):

    # extract affine
    sz_ddf = input_ddf.get_shape().as_list()
    # mbsz = mask_moving.shape[0].value
    grid1 = get_reference_grid([mask_motion.shape[i].value for i in [1, 2, 3]])
    pts1 = [tf.boolean_mask(grid1, mask_motion[i, :, :, :, 0] >= 0.5) for i in range(sz_ddf[0])]
    grid0 = grid1 + input_ddf
    pts0 = [tf.boolean_mask(grid0[i,:,:,:], mask_motion[i,:,:,:,0] >= 0.5) for i in range(sz_ddf[0])]
    affine = [tf.matrix_solve_ls(
        tf.concat([pts0[i], tf.ones_like(pts0[i][:, :1])], axis=1),
        tf.concat([pts1[i], tf.ones_like(pts1[i][:, :1])], axis=1)) for i in range(sz_ddf[0])]
    grid_d = tf.stack([tf.reshape(
        tf.matmul(tf.concat([tf.reshape(grid0[i,:,:,:],[-1,3]), tf.ones([sz_ddf[1]*sz_ddf[2]*sz_ddf[3],1])], axis=1),
                  affine[i])[:,0:3], sz_ddf[1:5]) for i in range(sz_ddf[0])], axis=0)
    def_ddf = grid_d - grid1

    len_target1, cen_target1, grid_target1 = compute_bounding_box(mask_target)
    len_motion1, cen_motion1, _ = compute_bounding_box(mask_motion)  # deformed
    # deformation
    s_t1, t_t1 = compute_st(cen_target1, cen_motion1, len_target1, len_motion1)
    grid_sample = grid_target1 * s_t1 + t_t1

    resampled_ddf = resample_linear(def_ddf, grid_sample) / scale_ratio
    return resampled_ddf


def normalise_ddf_spatial(mask_moving, mask_target, mask_motion, input_ddf, scale_ratio=1):

    len_moving0, cen_moving0, _ = compute_bounding_box(mask_moving)
    len_target1, cen_target1, grid_target1 = compute_bounding_box(mask_target)
    len_motion1, cen_motion1, _ = compute_bounding_box(mask_motion)  # deformed
    len_motion0, cen_motion0, _ = compute_bounding_box(mask_motion, input_ddf)  # un-deformed
    # change-of-coordinate
    s_t1, t_t1 = compute_st(cen_motion1, cen_target1, len_motion1, len_target1)
    s_10, t_10 = compute_st(cen_motion0, cen_motion1, len_motion0, len_motion1)
    s_t10, t_t10 = combine_st(s_t1, t_t1, s_10, t_10)
    s_1t, t_1t = compute_st(cen_target1, cen_motion1, len_target1, len_motion1)  # use the deformed
    s_t10t, t_t10t = combine_st(s_t10, t_t10, s_1t, t_1t)
    len_target0 = len_target1*s_t10t
    cen_target0 = cen_target1*s_t10t + t_t10t
    s_tm, t_tm = compute_st(cen_target0, cen_moving0, len_target0, len_moving0)
    # grid_target1 = tf.tile(tf.expand_dims(grid_target1, axis=0), [mbsz0, 1, 1, 1, 1])
    grid_moving1 = grid_target1 * s_tm + t_tm
    # deformation
    s_t1, t_t1 = compute_st(cen_target1, cen_motion1, len_target1, len_motion1)
    grid_sample = grid_target1 * s_t1 + t_t1

    resampled_ddf = resample_linear(input_ddf, grid_sample) / scale_ratio
    return grid_moving1 - grid_target1 + resampled_ddf


def normalise_ddf_magnitude(input_ddf):
    return input_ddf / tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(input_ddf), axis=4, keep_dims=True)), axis=[1,2,3], keep_dims=True)


def discriminator_regularizer(D1_logits, D1_arg, D2_logits, D2_arg, eps=1e-6):

    batch_size = D1_logits.shape[0].value
    D1 = tf.nn.sigmoid(D1_logits)
    D2 = tf.nn.sigmoid(D2_logits)
    grad_D1_logits = tf.gradients(D1_logits, D1_arg)[0]
    grad_D2_logits = tf.gradients(D2_logits, D2_arg)[0]
    grad_D1_logits_norm = tf.norm(tf.reshape(grad_D1_logits, [batch_size,-1]), axis=1, keep_dims=True)
    grad_D2_logits_norm = tf.norm(tf.reshape(grad_D2_logits, [batch_size,-1]), axis=1, keep_dims=True)
    # grad_D1_logits_norm = tf.reduce_mean(tf.square(tf.reshape(grad_D1_logits, [batch_size,-1])), axis=1, keep_dims=True)
    # grad_D2_logits_norm = tf.reduce_mean(tf.square(tf.reshape(grad_D2_logits, [batch_size,-1])), axis=1, keep_dims=True)

    reg_D1 = tf.multiply(tf.square(1.0-D1), tf.square(grad_D1_logits_norm))
    reg_D2 = tf.multiply(tf.square(D2), tf.square(grad_D2_logits_norm))
    disc_regularizer = tf.reduce_mean(reg_D1 + reg_D2)
    return disc_regularizer

""" temp for debug
D1 = tf.nn.sigmoid(logits_real)
D2 = tf.nn.sigmoid(logits_fake)
reg_D1 = tf.multiply(tf.square(1.0 - D1), tf.square(tf.norm(tf.reshape(tf.gradients(logits_real, motionDDF_norm_feature)[0], [miniBatchSize, -1]), axis=1, keep_dims=True)))
reg_D2 = tf.multiply(tf.square(D2), tf.square(tf.norm(tf.reshape(tf.gradients(logits_fake, displacement_feature)[0], [miniBatchSize, -1]), axis=1, keep_dims=True)))
loss_d += tf.reduce_mean(reg_D1 + reg_D2) * 0  # (gamma_disc_ph / 2.0)
"""


# utility nodes - before filtering
# movingLabel_warped0 = tftools.resample_linear(movingLabel0, grid_sample_global + displacement_local)
dice, movingVol, targetVol = tftools.compute_dice(movingLabel_warped0, targetLabel0)
dist = tftools.compute_centroid_distance(movingLabel_warped0, targetLabel0, grid_reference)
movingLabel_warped_global = tftools.resample_linear(movingLabel0, grid_sample_global)  # TODO: bug here - needs to be add dimension to grid_sample_global if needed
if (network_type == 'pre-warp') | (network_type == 'two-stream'):
    dice_global, movingVol_global, targetVol_global = tftools.compute_dice(movingLabel_warped_global, targetLabel0)
    dist_global = tftools.compute_centroid_distance(movingLabel_warped_global, targetLabel0, grid_reference)

