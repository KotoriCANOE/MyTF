import re
import numpy as np
import tensorflow as tf
from utils import helper
import utils.image

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

def activation_summary(x):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.
    Args:
        x: Tensor
    Returns:
        nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('{}_[0-9]*/'.format(TOWER_NAME), '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def loss_summaries(losses, decay=0):
    if decay > 0:
        loss_averages = tf.train.ExponentialMovingAverage(decay, name='loss_moving_average')
        loss_averages_op = loss_averages.apply(losses)
    else:
        loss_averages_op = None
    
    for l in losses:
        tf.summary.scalar(l.op.name + '.raw', l)
        if loss_averages_op:
            tf.summary.scalar(l.op.name, loss_averages.average(l))
    return loss_averages_op

def get_variable(name, shape, initializer, collection=None, trainable=True):
    """Helper to create a Variable stored on CPU memory.
    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable
    Returns:
      Variable Tensor
    """
    var = tf.get_variable(name, shape, dtype=tf.float32,
                          initializer=initializer,
                          trainable=trainable)
    # Add variable to collection
    if collection:
        tf.add_to_collection(collection, var)
    return var

def conv2d_variable(name, shape, initializer, init_factor=None, wd=None, collection=None, trainable=True):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
        name: name of the variable
        shape: list of ints
        stddev: standard deviation of a truncated Gaussian
        wd: add L2Loss weight decay multiplied by this float. If None, weight
            decay is not added for this Variable.
    Returns:
        Variable Tensor
    """
    dtype = tf.float32
    shape = helper.dim2int(shape)
    # weights initializer
    if init_factor is None:
        init_factor = 2.0 if initializer == 4 else 1.0
    if initializer == 1: # uniform Xavier initializer
        initializer = tf.variance_scaling_initializer(
            scale=init_factor, mode='fan_avg', distribution='uniform')
    elif initializer == 2: # normal Xavier initializer
        initializer = tf.variance_scaling_initializer(
            scale=init_factor, mode='fan_avg', distribution='normal')
    elif initializer == 3: # Convolutional Architecture for Fast Feature Embedding
        initializer = tf.variance_scaling_initializer(
            scale=init_factor, mode='fan_in', distribution='uniform')
    elif initializer == 4: # Delving Deep into Rectifiers, init_factor should be 2.0 for ReLU
        initializer = tf.variance_scaling_initializer(
            scale=init_factor, mode='fan_in', distribution='normal')
    elif initializer >= 5: # modified Xavier initializer
        stddev = np.sqrt(init_factor / (np.sqrt(shape[2] * shape[3]) * shape[0] * shape[1]))
        initializer = tf.truncated_normal_initializer(stddev=stddev, dtype=dtype)
    # weights initialization
    var = get_variable(name, shape, initializer, collection, trainable)
    # L2 regularization (weight decay)
    if wd is not None and wd != 0:
        regularizer = tf.contrib.layers.l2_regularizer(wd)
        regularizer(var)
    return var

def apply_batch_norm(last, decay=0.999, is_training=False, data_format='NHWC',
                     collection=None):
    if decay:
        return tf.contrib.layers.batch_norm(last, decay=decay, center=True, scale=True,
            fused=True, is_training=is_training, variables_collections=collection,
            data_format=data_format)
    else:
        return last

def PReLU(last, data_format='NHWC'):
    shape = last.get_shape()
    shape = shape[-3] if data_format == 'NCHW' else shape[-1]
    shape = [shape, 1, 1]
    alpha = get_variable('alpha', shape, tf.zeros_initializer())
    if data_format != 'NCHW':
        alpha = tf.squeeze(alpha, axis=[-2, -1])
        #alpha = tf.expand_dims(tf.expand_dims(alpha, -1), -1)
    return tf.maximum(0.0, last) + alpha * tf.minimum(0.0, last)

def apply_activation(last, activation, data_format='NHWC'):
    if isinstance(activation, str):
        activation = activation.lower()
    if activation and activation != 'none':
        if activation == 'relu':
            last = tf.nn.relu(last)
        elif activation == 'prelu':
            last = PReLU(last, data_format)
        elif activation[0:5] == 'lrelu':
            alpha = activation[5:]
            if alpha: alpha = float(alpha)
            else: alpha = 0.3
            lrelu = tf.contrib.keras.layers.LeakyReLU(alpha=alpha)
            last = lrelu(last)
        else:
            raise ValueError('Unrecognized \'activation\' specified!')
        #activation_summary(last)
    return last

def conv2d(last, ksize, out_channels,
           stride=1, padding='SAME', data_format='NHWC',
           batch_norm=None, is_training=False, activation=None,
           initializer=1, init_factor=1.0, wd=None, collection=None):
    # parameters
    in_channels = last.get_shape()[-3] if data_format == 'NCHW' else last.get_shape()[-1]
    if isinstance(ksize, int) or isinstance(ksize, tf.Dimension):
        ksize = [ksize, ksize]
    if isinstance(stride, int) or isinstance(stride, tf.Dimension):
        stride = [1, 1, stride, stride] if data_format == 'NCHW' else [1, stride, stride, 1]
    # convolution 2D
    kshape = [ksize[0], ksize[1], in_channels, out_channels]
    kernel = conv2d_variable('weights', shape=kshape,
                             initializer=initializer, init_factor=init_factor,
                             wd=wd, collection=collection)
    last = tf.nn.conv2d(last, kernel, strides=stride,
                        padding=padding, data_format=data_format)
    biases = get_variable('biases', [out_channels], tf.zeros_initializer())
    last = tf.nn.bias_add(last, biases, data_format=data_format)
    # batch normalization
    last = apply_batch_norm(last, decay=batch_norm, is_training=is_training,
        data_format=data_format)
    # activation function
    last = apply_activation(last, activation, data_format)
    return last

def depthwise_conv2d(last, ksize, channel_multiplier=1,
                     stride=1, padding='SAME', data_format='NHWC',
                     batch_norm=None, is_training=False, activation=None,
                     initializer=1, init_factor=1.0, wd=None, collection=None):
    # parameters
    in_channels = last.get_shape()[-3] if data_format == 'NCHW' else last.get_shape()[-1]
    out_channels = in_channels * channel_multiplier
    if isinstance(ksize, int) or isinstance(ksize, tf.Dimension):
        ksize = [ksize, ksize]
    if isinstance(stride, int) or isinstance(stride, tf.Dimension):
        stride = [1, 1, stride, stride] if data_format == 'NCHW' else [1, stride, stride, 1]
    # convolution 2D
    depthwise_kshape = [ksize[0], ksize[1], in_channels, channel_multiplier]
    depthwise_kernel = conv2d_variable('depthwise_weights', shape=depthwise_kshape,
                                       initializer=initializer, init_factor=init_factor,
                                       wd=wd, collection=collection)
    last = tf.nn.depthwise_conv2d_native(last, depthwise_kernel, strides=stride,
                                         padding=padding, data_format=data_format)
    biases = get_variable('biases', [out_channels], tf.zeros_initializer())
    last = tf.nn.bias_add(last, biases, data_format=data_format)
    # batch normalization
    last = apply_batch_norm(last, decay=batch_norm, is_training=is_training,
        data_format=data_format)
    # activation function
    last = apply_activation(last, activation, data_format)
    return last

def separable_conv2d(last, ksize, channel_multiplier=1, out_channels=None,
                     stride=1, padding='SAME', data_format='NHWC',
                     batch_norm=None, is_training=False, activation=None,
                     initializer=1, init_factor=1.0, wd=None, collection=None):
    # parameters
    in_channels = last.get_shape()[-3] if data_format == 'NCHW' else last.get_shape()[-1]
    temp_channels = in_channels * channel_multiplier
    if out_channels is None: out_channels = temp_channels
    if isinstance(ksize, int) or isinstance(ksize, tf.Dimension):
        ksize = [ksize, ksize]
    if isinstance(stride, int) or isinstance(stride, tf.Dimension):
        stride = [1, 1, stride, stride] if data_format == 'NCHW' else [1, stride, stride, 1]
    # convolution 2D
    depthwise_kshape = [ksize[0], ksize[1], in_channels, channel_multiplier]
    pointwise_kshape = [1, 1, temp_channels, out_channels]
    depthwise_kernel = conv2d_variable('depthwise_weights', shape=depthwise_kshape,
                                       initializer=initializer, init_factor=init_factor,
                                       wd=wd, collection=collection)
    pointwise_kernel = conv2d_variable('pointwise_weights', shape=pointwise_kshape,
                                       initializer=initializer, init_factor=init_factor,
                                       wd=wd, collection=collection)
    last = tf.nn.separable_conv2d(last, depthwise_kernel, pointwise_kernel, strides=stride,
                                  padding=padding, data_format=data_format)
    biases = get_variable('biases', [out_channels], tf.zeros_initializer())
    last = tf.nn.bias_add(last, biases, data_format=data_format)
    # batch normalization
    last = apply_batch_norm(last, decay=batch_norm, is_training=is_training,
        data_format=data_format)
    # activation function
    last = apply_activation(last, activation, data_format)
    return last

# checkerboard artifacts free resize convolution
# https://distill.pub/2016/deconv-checkerboard/
def resize_conv2d(last, ksize, out_channels,
                  scaling=2, data_format='NHWC',
                  batch_norm=None, is_training=False, activation=None,
                  initializer=1, init_factor=1.0, wd=None, collection=None):
    # parameters
    in_channels = last.get_shape()[-3] if data_format == 'NCHW' else last.get_shape()[-1]
    if isinstance(ksize, int) or isinstance(ksize, tf.Dimension):
        ksize = [ksize, ksize]
    if isinstance(scaling, int) or isinstance(scaling, tf.Dimension):
        scaling = [scaling, scaling]
    shape = tf.shape(last)
    if data_format == 'NCHW':
        out_size = [shape[2] * scaling[0], shape[3] * scaling[1]]
        out_shape = [shape[0], out_channels, out_size[0], out_size[1]]
    else:
        out_size = [shape[1] * scaling[0], shape[2] * scaling[1]]
        out_shape = [shape[0], out_size[0], out_size[1], out_channels]
    # nearest-neighbor upsample
    if data_format == 'NCHW':
        last = utils.image.NCHW2NHWC(last)
    last = tf.image.resize_nearest_neighbor(last, size=out_size)
    if data_format == 'NCHW':
        last = utils.image.NHWC2NCHW(last)
    # deconvolution 2D
    '''
    kshape = [ksize[0], ksize[1], in_channels, out_channels]
    kernel = conv2d_variable('weights', shape=kshape,
                             initializer=initializer, init_factor=init_factor,
                             wd=wd, collection=collection)
    last = tf.nn.conv2d(last, kernel, strides=[1, 1, 1, 1],
                        padding='SAME', data_format=data_format)
    '''
    kshape = [ksize[0], ksize[1], out_channels, in_channels]
    kernel = conv2d_variable('weights', shape=kshape,
                             initializer=initializer, init_factor=init_factor,
                             wd=wd, collection=collection)
    last = tf.nn.conv2d_transpose(last, kernel, out_shape, strides=[1, 1, 1, 1],
                                  padding='SAME', data_format=data_format)
    biases = get_variable('biases', [out_channels], tf.zeros_initializer())
    last = tf.nn.bias_add(last, biases, data_format=data_format)
    # batch normalization
    last = apply_batch_norm(last, decay=batch_norm, is_training=is_training,
        data_format=data_format)
    # activation function
    last = apply_activation(last, activation, data_format)
    return last

def depthwise_resize_conv2d(last, ksize, channel_multiplier=1,
                            scaling=2, data_format='NHWC',
                            batch_norm=None, is_training=False, activation=None,
                            initializer=1, init_factor=1.0, wd=None, collection=None):
    # parameters
    in_channels = last.get_shape()[-3] if data_format == 'NCHW' else last.get_shape()[-1]
    out_channels = in_channels // channel_multiplier
    if isinstance(ksize, int) or isinstance(ksize, tf.Dimension):
        ksize = [ksize, ksize]
    if isinstance(scaling, int) or isinstance(scaling, tf.Dimension):
        scaling = [scaling, scaling]
    shape = tf.shape(last)
    if data_format == 'NCHW':
        out_size = [shape[2] * scaling[0], shape[3] * scaling[1]]
        out_shape = [shape[0], out_channels, out_size[0], out_size[1]]
    else:
        out_size = [shape[1] * scaling[0], shape[2] * scaling[1]]
        out_shape = [shape[0], out_size[0], out_size[1], out_channels]
    # nearest-neighbor upsample
    if data_format == 'NCHW':
        last = utils.image.NCHW2NHWC(last)
    last = tf.image.resize_nearest_neighbor(last, size=out_size)
    if data_format == 'NCHW':
        last = utils.image.NHWC2NCHW(last)
    # deconvolution 2D
    depthwise_kshape = [ksize[0], ksize[1], in_channels, channel_multiplier]
    depthwise_kernel = conv2d_variable('depthwise_weights', shape=depthwise_kshape,
                                       initializer=initializer, init_factor=init_factor,
                                       wd=wd, collection=collection)
    '''
    last = tf.nn.depthwise_conv2d_native(last, depthwise_kernel, strides=[1, 1, 1, 1],
                                         padding='SAME', data_format=data_format)
    '''
    last = tf.nn.depthwise_conv2d_native_backprop_input(out_shape, depthwise_kernel, last,
            strides=[1, 1, 1, 1], padding='SAME', data_format=data_format)
    biases = get_variable('biases', [out_channels], tf.zeros_initializer())
    last = tf.nn.bias_add(last, biases, data_format=data_format)
    # batch normalization
    last = apply_batch_norm(last, decay=batch_norm, is_training=is_training,
        data_format=data_format)
    # activation function
    last = apply_activation(last, activation, data_format)
    return last

# implementation of Periodic Shuffling for sub-pixel convolution
# https://github.com/Tetrachrome/subpixel
def _phase_shift(I, r, shape, data_format='NHWC'):
    if data_format == 'NCHW':
        N, H, W = tf.shape(I)[0], shape[2], shape[3]
        X = tf.reshape(I, (N, r[0], r[1], H, W)) # N, rH, rW, H, W
        X = tf.split(X, H, axis=-2) # H, [N, rH, rW, 1, W]
        X = [tf.reshape(x, (N, r[1], r[0], W)) for x in X] # H, [N, rH, rW, W]
        X = tf.concat(X, axis=-3) # N, H*rH, rW, W
        X = tf.split(X, W, axis=-1) # W, [N, H*rH, rW, 1]
        X = [tf.reshape(x, (N, H*r[0], r[1])) for x in X] # W, [N, H*rH, rW]
        X = tf.concat(X, axis=-1) # N, H*rH, W*rW
        return tf.reshape(X, (N, 1, H*r[0], W*r[1]))
    else:
        N, H, W = tf.shape(I)[0], shape[1], shape[2]
        X = tf.reshape(I, (N, H, W, r[0], r[1])) # N, H, W, rH, rW
        X = tf.split(X, H, axis=-4) # H, [N, 1, W, rH, rW]
        X = [tf.reshape(x, (N, W, r[1], r[0])) for x in X] # H, [N, W, rH, rW]
        X = tf.concat(X, axis=-2) # N, W, H*rH, rW
        X = tf.split(X, W, axis=-3) # W, [N, 1, H*rH, rW]
        X = [tf.reshape(x, (N, H*r[0], r[1])) for x in X] # W, [N, H*rH, rW]
        X = tf.concat(X, axis=-1) # N, H*rH, W*rW
        return tf.reshape(X, (N, H*r[0], W*r[1], 1))

def periodic_shuffling(X, r, data_format='NHWC'):
    # require statically known shape (None, H, W, C) or (None, C, H, W)
    if isinstance(r, int) or isinstance(r, tf.Dimension):
        r = [r, r]
    shape = helper.dim2int(X.get_shape())
    if data_format == 'NCHW':
        channels = shape[-3] // (r[0] * r[1])
        Xc = tf.split(X, channels, axis=-3)
        shape[-3] = r[0] * r[1]
        X = tf.concat([_phase_shift(x, r, shape, data_format) for x in Xc], axis=-3)
    else:
        channels = shape[-1] // (r[0] * r[1])
        Xc = tf.split(X, channels, axis=-1)
        shape[-1] = r[0] * r[1]
        X = tf.concat([_phase_shift(x, r, shape, data_format) for x in Xc], axis=-1)
    return X

def subpixel_conv2d(last, ksize, out_channels,
                    scaling=2, padding='SAME', data_format='NHWC',
                    batch_norm=None, is_training=False, activation=None,
                    initializer=1, init_factor=1.0, wd=None, collection=None):
    # parameters
    in_channels = last.get_shape()[-3] if data_format == 'NCHW' else last.get_shape()[-1]
    if isinstance(ksize, int) or isinstance(ksize, tf.Dimension):
        ksize = [ksize, ksize]
    if isinstance(scaling, int) or isinstance(scaling, tf.Dimension):
        scaling = [scaling, scaling]
    temp_channels = out_channels * scaling[0] * scaling[1]
    # convolution 2D
    kshape = [ksize[0], ksize[1], in_channels, temp_channels]
    kernel = conv2d_variable('weights', shape=kshape,
                             initializer=initializer, init_factor=init_factor,
                             wd=wd, collection=collection)
    last = tf.nn.conv2d(last, kernel, strides=[1, 1, 1, 1],
                        padding=padding, data_format=data_format)
    biases = get_variable('biases', [temp_channels], tf.zeros_initializer())
    last = tf.nn.bias_add(last, biases, data_format=data_format)
    # periodic shuffling
    last = periodic_shuffling(last, scaling, data_format)
    # batch normalization
    last = apply_batch_norm(last, decay=batch_norm, is_training=is_training,
        data_format=data_format)
    # activation function
    last = apply_activation(last, activation, data_format)
    return last

def depthwise_subpixel_conv2d(last, ksize, channel_multiplier=1,
                              scaling=2, padding='SAME', data_format='NHWC',
                              batch_norm=None, is_training=False, activation=None,
                              initializer=1, init_factor=1.0, wd=None, collection=None):
    # parameters
    in_channels = last.get_shape()[-3] if data_format == 'NCHW' else last.get_shape()[-1]
    if isinstance(ksize, int) or isinstance(ksize, tf.Dimension):
        ksize = [ksize, ksize]
    if isinstance(scaling, int) or isinstance(scaling, tf.Dimension):
        scaling = [scaling, scaling]
    channel_multiplier *= scaling[0] * scaling[1]
    out_channels = in_channels * channel_multiplier
    # convolution 2D
    depthwise_kshape = [ksize[0], ksize[1], in_channels, channel_multiplier]
    depthwise_kernel = conv2d_variable('depthwise_weights', shape=depthwise_kshape,
                                       initializer=initializer, init_factor=init_factor,
                                       wd=wd, collection=collection)
    last = tf.nn.depthwise_conv2d_native(last, depthwise_kernel, strides=[1, 1, 1, 1],
                                         padding=padding, data_format=data_format)
    biases = get_variable('biases', [out_channels], tf.zeros_initializer())
    last = tf.nn.bias_add(last, biases, data_format=data_format)
    # periodic shuffling
    last = periodic_shuffling(last, scaling, data_format)
    # batch normalization
    last = apply_batch_norm(last, decay=batch_norm, is_training=is_training,
        data_format=data_format)
    # activation function
    last = apply_activation(last, activation, data_format)
    return last
