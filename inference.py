import tensorflow as tf

import config
import labelreg.helpers as helper
import labelreg.networks as network


# 1 - images to register
reader_moving_image, reader_fixed_image, _, _ = helper.get_data_readers(config.Inference.dir_moving_image,
                                                                        config.Inference.dir_fixed_image)

# 2 - graph for predicting ddf only
ph_moving_image = tf.placeholder(tf.float32, [reader_moving_image.num_data]+reader_moving_image.data_shape+[1])
ph_fixed_image = tf.placeholder(tf.float32, [reader_fixed_image.num_data]+reader_fixed_image.data_shape+[1])

reg_net = network.build_network(network_type=config.Network.network_type,
                                minibatch_size=reader_moving_image.num_data,
                                image_moving=ph_moving_image,
                                image_fixed=ph_fixed_image)

# restore the trained weights
saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, config.Inference.file_model_saved)


# compute ddf for resampling
testFeed = {ph_moving_image: reader_moving_image.get_data(),
            ph_fixed_image: reader_fixed_image.get_data()}
ddf = sess.run(reg_net.ddf, feed_dict=testFeed)

if config.Inference.file_ddf_save is not None:
    import nibabel as nib
    affine = [[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]]
    [nib.save(nib.Nifti1Image(ddf[idx, ...], affine), 'ddf_%s.nii' % idx)
     for idx in range(reader_moving_image.num_data)]