import tensorflow as tf

# variables
def var_conv_kernel(ch_in, ch_out, k_conv=None, initialiser=None):
    if k_conv is None:
        k_conv = [3, 3, 3]
    if initialiser is None:
        initialiser = tf.contrib.layers.xavier_initializer()
    return tf.get_variable("W", shape=k_conv+[ch_in]+[ch_out], initializer=initialiser)


def var_bias(b_shape, initialiser=None):
    if initialiser is None:
        initialiser = tf.contrib.layers.xavier_initializer()
    return tf.get_variable("b", shape=[b_shape], initializer=initialiser)


# blocks
def conv3_block(input, w, strides=None):
    if strides is None:
        strides = [1, 2, 2, 2, 1]
    return tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(input, w, strides, "SAME")))


def deconv3_block(input, w, shape_out, strides):
    return tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d_transpose(input, w, shape_out, strides, "SAME")))


def downsample_resnet_block(input, ch_in, ch_out, k_conv0=None, use_pooling=True, name='downsample_resnet_block'):
    if k_conv0 is None:
        k_conv0 = [3, 3, 3]
    strides1 = [1, 1, 1, 1, 1]
    strides2 = [1, 2, 2, 2, 1]
    with tf.variable_scope(name):
        w0 = var_conv_kernel(ch_in, ch_out, k_conv0)
        wr1 = var_conv_kernel(ch_out, ch_out)
        wr2 = var_conv_kernel(ch_out, ch_out)
        h0 = conv3_block(input, w0, strides1)
        r1 = conv3_block(h0, wr1, strides1)
        r2 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(r1, wr2, strides1, "SAME")) + h0)
        if use_pooling:
            k_pool = [1, 2, 2, 2, 1]
            h1 = tf.nn.max_pool3d(r2, k_pool, strides2, padding="SAME")
        else:
            w1 = var_conv_kernel(ch_out, ch_out)
            h1 = conv3_block(r2, w1, strides2)
        return h1, h0


def upsample_resnet_block(input, input_skip, ch_in, ch_out, use_additive_upsampling=True, name='upsample_resnet_block'):
    strides1 = [1, 1, 1, 1, 1]
    strides2 = [1, 2, 2, 2, 1]
    size_out = input_skip.get_shape()
    with tf.variable_scope(name):
        w0 = var_conv_kernel(ch_out, ch_in)
        wr1 = var_conv_kernel(ch_out, ch_out)
        wr2 = var_conv_kernel(ch_out, ch_out)
        h0 = deconv3_block(input, w0, size_out, strides2)
        if use_additive_upsampling:
            h0 += additive_up_sampling(input, size_out)
        r1 = h0 + input_skip
        r2 = conv3_block(h0, wr1, strides1, "SAME")
        h1 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(r1, wr2, strides1, "SAME")) + r1)
        return h1


def ddf_summand(input, w, b, shape_out):
    strides1 = [1, 1, 1, 1, 1]
    if input.get_shape() == shape_out:
        return tf.nn.conv3d(input, w, strides1, "SAME") + b
    else:
        return resize_volume(tf.nn.conv3d(input, w, strides1, "SAME") + b, shape_out)


# layers
def additive_up_sampling(layer, size, stride=2, name='additive_upsampling'):
    with tf.variable_scope(name):
        return tf.reduce_sum(tf.stack(tf.split(resize_volume(layer, size), stride, axis=4), axis=5), axis=5)


def resize_volume(image, size, method=0, name='resize_volume'):
    # size is [depth, height width]
    # image is Tensor with shape [batch, depth, height, width, channels]
    with tf.variable_scope(name):
        reshaped2d = tf.reshape(image, [-1, int(image.get_shape()[2]), int(image.get_shape()[3]), int(image.get_shape()[4])])
        resized2d = tf.image.resize_images(reshaped2d,[size[1],size[2]],method)
        reshaped2d = tf.reshape(resized2d, [int(image.get_shape()[0]), int(image.get_shape()[1]), size[1], size[2], int(image.get_shape()[4])])
        permuted = tf.transpose(reshaped2d, [0,3,2,1,4])
        reshaped2db = tf.reshape(permuted, [-1, size[1], int(image.get_shape()[1]), int(image.get_shape()[4])])
        resized2db = tf.image.resize_images(reshaped2db,[size[1],size[0]],method)
        reshaped2db = tf.reshape(resized2db, [int(image.get_shape()[0]), size[2], size[1], size[0], int(image.get_shape()[4])])
        return tf.transpose(reshaped2db, [0, 3, 2, 1, 4])