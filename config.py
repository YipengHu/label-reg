import os


class Data:
    # data
    dir_moving_image = os.path.join(os.environ['HOME'], 'git/label-reg-demo/data/mr_images')
    dir_fixed_image = os.path.join(os.environ['HOME'], 'git/label-reg-demo/data/us_images')
    dir_moving_label = os.path.join(os.environ['HOME'], 'git/label-reg-demo/data/mr_labels')
    dir_fixed_label = os.path.join(os.environ['HOME'], 'git/label-reg-demo/data/us_labels')


class Network:
    # network
    network_type = 'local'  # 'global', 'local', 'composite'
    ddf_levels = [0, 1, 2, 3, 4]


class Loss:
    similarity_type = 'dice'  # 'cross-entropy'
    similarity_scales = [0, 1, 2, 4, 8, 16]
    regulariser_type = 'bending'  # 'gradient-l2'
    regulariser_weight = 0.5


class Train:
    total_iterations = int(1e5)
    minibatch_size = 8
    learning_rate = 1e-5
    # output
    info_print_freq = 2
    dir_model_save = os.path.join(os.environ['HOME'], 'git/label-reg-demo/model')
