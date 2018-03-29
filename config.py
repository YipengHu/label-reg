import os


class Data:
    dir_moving_image = os.path.join(os.environ['HOME'], 'git/label-reg-demo/data/train/mr_images')
    dir_fixed_image = os.path.join(os.environ['HOME'], 'git/label-reg-demo/data/train/us_images')
    dir_moving_label = os.path.join(os.environ['HOME'], 'git/label-reg-demo/data/train/mr_labels')
    dir_fixed_label = os.path.join(os.environ['HOME'], 'git/label-reg-demo/data/train/us_labels')


class Network:
    network_type = 'local'  # 'global', 'local', 'composite'


class Loss:
    similarity_type = 'dice'  # 'cross-entropy'
    similarity_scales = [0, 1, 2, 4, 8, 16]  # smaller variances for example data
    regulariser_type = 'bending'  # 'gradient-l2'
    regulariser_weight = 0.5


class Train:
    total_iterations = int(1e5)
    minibatch_size = 4
    learning_rate = 1e-5
    # output
    freq_info_print = 100
    freq_model_save = 500
    file_model_save = os.path.join(os.environ['HOME'], 'git/label-reg-demo/data/model.ckpt')


class Inference:
    file_model_saved = Train.file_model_save
    dir_moving_image = os.path.join(os.environ['HOME'], 'git/label-reg-demo/data/test/mr_images')
    dir_fixed_image = os.path.join(os.environ['HOME'], 'git/label-reg-demo/data/test/us_images')
    dir_save = os.path.join(os.environ['HOME'], 'git/label-reg-demo/data/')
    # for testing only
    dir_moving_label = os.path.join(os.environ['HOME'], 'git/label-reg-demo/data/test/mr_labels')
    dir_fixed_label = os.path.join(os.environ['HOME'], 'git/label-reg-demo/data/test/us_labels')

