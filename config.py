import os


class Data:
    dir_moving_image = os.path.join(os.environ['HOME'], 'git/label-reg-demo/data/mr_images')
    dir_fixed_image = os.path.join(os.environ['HOME'], 'git/label-reg-demo/data/us_images')
    dir_moving_label = os.path.join(os.environ['HOME'], 'git/label-reg-demo/data/mr_labels')
    dir_fixed_label = os.path.join(os.environ['HOME'], 'git/label-reg-demo/data/us_labels')


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
    info_print_freq = 100
    dir_model_save = os.path.join(os.environ['HOME'], 'git/label-reg-demo/model')
