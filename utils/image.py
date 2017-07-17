import numpy as np
import tensorflow as tf
from utils import helper

def RGB2OPP(images, norm=False, scope='RGB2OPP'):
    shape = images.get_shape()
    shape = helper.dim2int(shape)
    assert shape[-1] == 3
    last = images
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
        pixels = 1
        for s in shape[:-1]: pixels *= s
        last = tf.reshape(last, [pixels, shape[-1]])
        last = tf.matmul(last, t, transpose_b=True)
        last = tf.reshape(last, shape)
    return last

def OPP2RGB(images, norm=False, scope='OPP2RGB'):
    shape = images.get_shape()
    shape = helper.dim2int(shape)
    assert shape[-1] == 3
    last = images
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
        pixels = 1
        for s in shape[:-1]: pixels *= s
        last = tf.reshape(last, [pixels, shape[-1]])
        last = tf.matmul(last, t, transpose_b=True)
        last = tf.reshape(last, shape)
    return last
