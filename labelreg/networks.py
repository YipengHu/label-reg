import tensorflow as tf
import labelreg.layers as layer
import labelreg.utils as util
import labelreg.helpers as helper


def build_network(network_type, **kwargs):
    type_lower = network_type.lower()
    if type_lower == 'local':
        return LocalNet(**kwargs)
    elif type_lower == 'global':
        return GlobalNet(**kwargs)
    elif type_lower == 'composite':
        return CompositeNet(**kwargs)


class BaseNet:
    def __init__(self, minibatch_size, image_moving, image_fixed):
        self.minibatch_size = minibatch_size
        self.image_size = image_fixed.shape.as_list()[1:4]
        self.input_layer = tf.concat([layer.resize_volume(image_moving, self.image_size), image_fixed], axis=4)


class LocalNet(BaseNet):

    def __init__(self, ddf_levels, **kwargs):
        BaseNet.__init__(self, **kwargs)
        self.ddf_levels = ddf_levels
        self.num_channel_initial = 32  # defaults

        # build_graph:
        nc = [int(self.num_channel_initial*(2**i)) for i in range(5)]
        h0, hc0 = layer.downsample_resnet_block(self.input_layer, 2, nc[0], k_conv0=[7, 7, 7], name='down_0')
        h1, hc1 = layer.downsample_resnet_block(h0, nc[0], nc[1], name='down_1')
        h2, hc2 = layer.downsample_resnet_block(h1, nc[1], nc[2], name='down_2')
        h3, hc3 = layer.downsample_resnet_block(h2, nc[2], nc[3], name='down_3')
        hm = [layer.conv3_block(h3, nc[3], nc[4], name='deep_4')]

        min_level = min(self.ddf_levels)
        hm += [layer.upsample_resnet_block(hm[0], hc3, nc[4], nc[3], name='up_3')] if min_level < 4 else []
        hm += [layer.upsample_resnet_block(hm[1], hc2, nc[3], nc[2], name='up_2')] if min_level < 3 else []
        hm += [layer.upsample_resnet_block(hm[2], hc1, nc[2], nc[1], name='up_1')] if min_level < 2 else []
        hm += [layer.upsample_resnet_block(hm[3], hc0, nc[1], nc[0], name='up_0')] if min_level < 1 else []

        self.ddf = tf.reduce_sum(tf.stack([layer.ddf_summand(hm[4-idx], nc[idx], self.image_size, name='sum_%d' % idx)
                                           for idx in self.ddf_levels],
                                          axis=5), axis=5)

    def warp_image(self, input_image):
        return util.resample_linear(input_image, self.ddf)


class GlobalNet(BaseNet):
    def __init__(self, **kwargs):
        BaseNet.__init__(self, **kwargs)
        self.num_channel_initial_global = 8
        initial_bias_global = 0.0
        initial_std_global = 0.0


class CompositeNet(BaseNet):
    def __init__(self, **kwargs):
        BaseNet.__init__(self, **kwargs)
        global_net = GlobalNet(**kwargs)
        local_net = LocalNet(**kwargs)
