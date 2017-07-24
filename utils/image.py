import numpy as np
import tensorflow as tf
from utils import helper

def RGB2Y(images, scope='RGB2Y'):
    with tf.variable_scope(scope):
        c1 = 1 / 3
        coef = [c1,c1,c1]
        t = tf.constant(coef, shape=[1, 3], dtype=images.dtype)
        shape = tf.shape(images)
        with tf.control_dependencies([tf.assert_equal(shape[-1], 3)]):
            last = images
            last = tf.reshape(last, [tf.reduce_prod(shape[:-1]), 3])
            last = tf.matmul(last, t, transpose_b=True)
            shape = tf.concat([shape[:-1], [1]], axis=0)
            last = tf.reshape(last, shape)
    return last

def RGB2OPP(images, norm=False, scope='RGB2OPP'):
    with tf.variable_scope(scope):
        if norm:
            c1 = 1 / 3
            c2 = 1 / 2
            c3 = 1 / 4
            c4 = 1 / 2 # c3 * 2
        else:
            c1 = 1 / 3
            c2 = 1 / np.sqrt(6)
            c3 = np.sqrt(2) / 6
            c4 = np.sqrt(2) / 3 # c3 * 2
        coef = [c1,c1,c1, c2,0,-c2, c3,-c4,c3]
        t = tf.constant(coef, shape=[3, 3], dtype=images.dtype)
        shape = tf.shape(images)
        with tf.control_dependencies([tf.assert_equal(shape[-1], 3)]):
            last = images
            last = tf.reshape(last, [tf.reduce_prod(shape[:-1]), 3])
            last = tf.matmul(last, t, transpose_b=True)
            last = tf.reshape(last, shape)
    return last

def OPP2RGB(images, norm=False, scope='OPP2RGB'):
    with tf.variable_scope(scope):
        if norm:
            c1 = 1
            c2 = 1
            c3 = 2 / 3
            c4 = 4 / 3 # c3 * 2
        else:
            c1 = 1
            c2 = np.sqrt(6) / 2
            c3 = np.sqrt(2) / 2
            c4 = np.sqrt(2) # c3 * 2
        coef = [c1,c2,c3, c1,0,-c4, c1,-c2,c3]
        t = tf.constant(coef, shape=[3, 3], dtype=images.dtype)
        shape = tf.shape(images)
        with tf.control_dependencies([tf.assert_equal(shape[-1], 3)]):
            last = images
            last = tf.reshape(last, [tf.reduce_prod(shape[:-1]), 3])
            last = tf.matmul(last, t, transpose_b=True)
            last = tf.reshape(last, shape)
    return last

# SS-SSIM/MS-SSIM implementation
# https://github.com/tensorflow/models/blob/master/compression/image_encoder/msssim.py
# https://stackoverflow.com/a/39053516
def _fspecial_gauss(radius, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[1-radius:1+radius, 1-radius:1+radius]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp((x*x + y*y) / (-2.0*sigma*sigma))
    return g / tf.reduce_sum(g)

def SS_SSIM(img1, img2, ret_cs=False, mean_metric=True, radius=5, sigma=1.5, L=1):
    # L: depth of image (255 in case the image has a differnt scale)
    window = _fspecial_gauss(radius, sigma) # window shape [radius*2+1, radius*2+1]
    K1 = 0.01
    K2 = 0.03
    L_sq = L * L
    C1 = K1 * K1 * L_sq
    C2 = K2 * K2 * L_sq
    
    mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1], padding='VALID')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1], padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1], padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1], padding='VALID') - mu1_mu2
    l_map = (2.0 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
    cs_map = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    ssim_map = l_map * cs_map
    
    if mean_metric:
        ssim_map = tf.reduce_mean(ssim_map)
        cs_map = tf.reduce_mean(cs_map)
    if ret_cs: value = (ssim_map, cs_map)
    else: value = ssim_map
    return value

def MS_SSIM(img1, img2, weights=None, radius=5, sigma=1.5, L=1):
    if not weights: weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    weights = tf.constant(weights, dtype=tf.float32)
    levels = weights.get_shape()[0].value
    mssim = []
    mcs = []
    for _ in range(levels):
        ssim, cs = SS_SSIM(img1, img2, ret_cs=True, mean_metric=True,
                           radius=radius, sigma=sigma, L=L)
        mssim.append(ssim)
        mcs.append(cs)
        filtered_im1 = tf.nn.avg_pool(img1, [1,2,2,1], [1,2,2,1], padding='SAME')
        filtered_im2 = tf.nn.avg_pool(img2, [1,2,2,1], [1,2,2,1], padding='SAME')
        img1 = filtered_im1
        img2 = filtered_im2

    # list to tensor of dim D+1
    mcs = tf.stack(mcs, axis=0)

    value = tf.reduce_prod(mcs[0:levels - 1] ** weights[0:levels - 1]) * \
                          (mssim[levels - 1] ** weights[levels - 1])

    return value

# arXiv 1511.08861
def MS_SSIM_2(img1, img2, radius=5, sigma=[0.5, 1, 2, 4, 8], L=1, norm=True):
    levels = len(sigma)
    mssim = []
    mcs = []
    for _ in range(levels):
        ssim, cs = SS_SSIM(img1, img2, ret_cs=True, mean_metric=False,
                           radius=radius, sigma=sigma[_], L=L)
        mssim.append(ssim)
        mcs.append(cs)

    # list to tensor of dim D+1
    mcs = tf.stack(mcs, axis=0)

    value = tf.reduce_prod(mcs[0:levels - 1], axis=0) * mssim[levels - 1]
    value = tf.reduce_mean(value)
    if norm: value **= 1.0 / levels

    return value
