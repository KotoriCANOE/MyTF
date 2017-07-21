import numpy as np
import tensorflow as tf
from utils import helper

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

# M-SSIM/MS-SSIM implementation by bsautermeister
# https://stackoverflow.com/a/39053516
def _fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp((x*x + y*y)/(-2.0*sigma*sigma))
    return g / tf.reduce_sum(g)

def M_SSIM(img1, img2, cs_map=False, mean_metric=True, size=11, sigma=1.5):
    window = _fspecial_gauss(size, sigma) # window shape [size, size]
    K1 = 0.01
    K2 = 0.03
    L = 1  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1],padding='VALID')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1],padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1],padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1],padding='VALID') - mu1_mu2
    if cs_map:
        value = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)),
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        value = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value

def MS_SSIM(img1, img2, mean_metric=True, level=5):
    weight = tf.constant([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=tf.float32)
    mssim = []
    mcs = []
    for l in range(level):
        ssim_map, cs_map = M_SSIM(img1, img2, cs_map=True, mean_metric=False)
        mssim.append(tf.reduce_mean(ssim_map))
        mcs.append(tf.reduce_mean(cs_map))
        filtered_im1 = tf.nn.avg_pool(img1, [1,2,2,1], [1,2,2,1], padding='SAME')
        filtered_im2 = tf.nn.avg_pool(img2, [1,2,2,1], [1,2,2,1], padding='SAME')
        img1 = filtered_im1
        img2 = filtered_im2

    # list to tensor of dim D+1
    mssim = tf.stack(mssim, axis=0)
    mcs = tf.stack(mcs, axis=0)

    value = (tf.reduce_prod(mcs[0:level-1]**weight[0:level-1])*
                            (mssim[level-1]**weight[level-1]))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value
