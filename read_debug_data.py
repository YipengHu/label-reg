import numpy as np
import matplotlib.pyplot as plt

debug_steps = [0, 1]


slice_m = 30
slice_f = 20
mb_size = 4

for step in debug_steps:
    ph_moving_image_debug = np.load('debug%d_ph_moving_image.npy' % step)
    ph_fixed_image_debug = np.load('debug%d_ph_fixed_image.npy' % step)
    input_moving_image_debug = np.load('debug%d_input_moving_image.npy' % step)
    input_fixed_image_debug = np.load('debug%d_input_fixed_image.npy' % step)
    warped_moving_label_debug = np.load('debug%d_warped_moving_label.npy' % step)
    warped_image_debug = np.load('debug%d_warped_image.npy' % step)

    for idx_mb in range(mb_size):
        plt.figure()  # original diff, current diff; ddf-x, ddf-y
        plt.subplot(3, 2, 1), plt.imshow(np.squeeze(ph_moving_image_debug[idx_mb,:,:,slice_m,:]), cmap='gray'), plt.title('ph - moving')
        plt.subplot(3, 2, 2), plt.imshow(np.squeeze(ph_fixed_image_debug[idx_mb,:,:,slice_m,:]), cmap='gray'), plt.title('ph - fixed')
        plt.subplot(3, 2, 3), plt.imshow(np.squeeze(input_moving_image_debug[idx_mb,:,:,slice_m,:]), cmap='gray'), plt.title('input - moving')
        plt.subplot(3, 2, 4), plt.imshow(np.squeeze(input_fixed_image_debug[idx_mb,:,:,slice_m,:]), cmap='gray'), plt.title('input - fixed')
        plt.subplot(3, 2, 5), plt.imshow(np.squeeze(warped_moving_label_debug[idx_mb,:,:,slice_m,:]), cmap='gray'), plt.title('warped - moving')
        plt.subplot(3, 2, 6), plt.imshow(np.squeeze(warped_image_debug[idx_mb,:,:,slice_m,:]), cmap='gray'), plt.title('Mwarped - moving')
        plt.title('Mini-batch No.%d' % idx_mb)
