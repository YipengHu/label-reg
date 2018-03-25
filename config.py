import os


class Data:
    # data
    dir_moving_image = os.path.join(os.environ['HOME'], 'git/label-reg-demo/data/mr_images')
    dir_fixed_image = os.path.join(os.environ['HOME'], 'git/label-reg-demo/data/us_images')
    dir_moving_label = os.path.join(os.environ['HOME'], 'git/label-reg-demo/data/mr_labels')
    dir_fixed_label = os.path.join(os.environ['HOME'], 'git/label-reg-demo/data/us_labels')

    # network
    num_channel_initial = 32
    conv_size_initial = 7
    network_type = 'local'  # 'global', 'local', 'composite'
    ddf_summands = [0, 1, 2, 3, 4]
    loss_type = 'dice'  # 'ce'
    loss_scales = [0, 1, 2, 4, 8, 16]
    loss_scale_kernel = 'gauss'
    regulariser_type = 'bending'  # 'gradient-l2'
    regulariser_weight = 0.5

    if not (network_type == 'local'):
        num_channel_initial_global = 8
        initial_bias_global = 0.0
        initial_std_global = 0.0

    # training
    total_iterations = 1e5
    minibatch_size = 8
    learning_rate = 1e-5

    # verbose
    info_print_freq = 100