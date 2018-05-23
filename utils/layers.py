import re
import numpy as np
import tensorflow as tf
from utils import helper
import utils.image

def convert_range(x, range_in, range_out, saturate=False):
    scale = (range_out[1] - range_out[0]) / (range_in[1] - range_in[0])
    bias = range_out[0] - range_in[0] * scale
    if scale != 1:
        scale = tf.constant(scale, x.dtype)
        y = x * scale
    if bias != 0:
        bias = tf.constant(bias, x.dtype)
        y += bias
    if saturate:
        y = tf.clip_by_value(y, range_out[0], range_out[1])
    return y

def quantize(x, range_in, range_out, saturate=False, dtype=tf.int32):
    graph = tf.get_default_graph()
    grad_map = {'Round': 'Identity'}
    with graph.gradient_override_map(grad_map):
        return tf.round(convert_range(x, range_in, range_out, saturate))

def histogram(x, range, bins, saturate=True, dtype=tf.float32):
    graph = tf.get_default_graph()
    grad_map = {'Floor': 'Identity', 'Cast': 'Identity'}
    with graph.gradient_override_map(grad_map):
        scale = bins / (range[1] - range[0])
        bias = -range[0] * scale
        scale = tf.constant(scale, x.dtype)
        bias = tf.constant(bias, x.dtype)
        indices = tf.floor(x * scale + bias)
        if saturate:
            indices = tf.clip_by_value(indices, 0, bins - 1)
        indices = tf.cast(indices, tf.int32)
        return tf.unsorted_segment_sum(tf.ones_like(indices, dtype=dtype), indices, bins)

def entropy(x, range, bins, base=None, saturate=True, mean=False,
            dtype=tf.float32, epsilon=1e-8):
    if base is None: base = bins
    x_num = tf.cast(tf.reduce_prod(tf.shape(x)), dtype)
    #range = tf.constant(range, dtype=x.dtype)
    #hist = tf.histogram_fixed_width(x, range, bins, dtype=dtype)
    hist = histogram(x, range, bins, saturate, dtype)
    probs = hist / x_num
    ents = probs * tf.log(probs + epsilon)
    if mean:
        ent = tf.reduce_mean(ents)
    else:
        ent = tf.reduce_sum(ents)
    ent *= -1 / np.log(base)
    return ent

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
    var = tf.get_variable(name, shape, dtype=tf.float32,
                          initializer=initializer,
                          trainable=trainable)
    # Add variable to collection
    if collection:
        tf.add_to_collection(collection, var)
    return var

def conv2d_variable(name, shape, initializer, init_factor=None, collection=None, trainable=True):
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
    return var

def dense(last, out_channels, use_bias=None, activation=None,
          initializer=1, init_factor=1.0, collection=None):
    # parameters
    data_format = 'NHWC'
    in_channels = last.get_shape()[-1]
    kshape = [in_channels, out_channels]
    # weight
    kernel = conv2d_variable('weights', shape=kshape,
                             initializer=initializer, init_factor=init_factor,
                             collection=collection)
    last = tf.matmul(last, kernel)
    # bias
    if use_bias is None:
        use_bias = True
    if use_bias:
        biases = get_variable('biases', [out_channels], tf.zeros_initializer())
        last = tf.nn.bias_add(last, biases, data_format=data_format)
    # activation function
    last = apply_activation(last, activation, data_format, collection)
    return last

def apply_batch_norm(last, decay=0.999, train=False, data_format='NHWC',
                     renorm=None, collection=None):
    renorm_decay = 0.99
    if renorm:
        if isinstance(renorm, (int, float)):
            renorm_decay = renorm
        renorm = True
    if decay:
        return tf.contrib.layers.batch_norm(last, decay=decay, center=True, scale=True,
            fused=True, is_training=train, variables_collections=collection,
            data_format=data_format, renorm=renorm, renorm_decay=renorm_decay)
        '''
        return tf.layers.batch_normalization(last, axis=1 if data_format == 'NCHW' else -1,
            momentum=decay, center=True, scale=True, training=train, trainable=True,
            reuse=None, renorm=renorm, renorm_momentum=renorm_decay, fused=True)
        '''
    else:
        return last

def PReLU(last, data_format='NHWC', collection=None):
    with tf.variable_scope('prelu') as scope:
        shape = last.get_shape()
        shape = shape[-3] if data_format == 'NCHW' else shape[-1]
        shape = [shape, 1, 1]
        alpha = get_variable('alpha', shape, tf.zeros_initializer(), collection)
        if data_format != 'NCHW':
            alpha = tf.squeeze(alpha, axis=[-2, -1])
        return tf.maximum(0.0, last) + alpha * tf.minimum(0.0, last)

def SelectionUnit(last, data_format='NHWC', collection=None):
    with tf.variable_scope('selection_unit') as scope:
        skip = last
        last = tf.nn.relu(last)
        with tf.variable_scope('pointwise_conv') as scope:
            last = conv2d(last, 1, None, use_bias=True,
                stride=1, padding='SAME', data_format=data_format,
                bn=None, train=False, activation='sigmoid',
                initializer=4, init_factor=2.0, collection=collection)
        return tf.multiply(skip, last)

def SqueezeExcitation(last, channels=None, channel_r=1, data_format='NHWC', collection=None):
    shape = helper.dim2int(last.get_shape())
    in_channels = shape[-3] if data_format == 'NCHW' else shape[-1]
    if channels is None: channels = in_channels
    channels //= channel_r
    with tf.variable_scope('squeeze_excitation') as scope:
        skip = last
        # global average pooling - NxC
        last = tf.reduce_mean(last, [-2, -1] if data_format == 'NCHW' else [-3, -2])
        # FC - Nx(C/r)
        with tf.variable_scope('dense1') as scope:
            last = dense(last, channels, use_bias=True, activation='relu',
              initializer=2, init_factor=1.0, collection=collection)
        # FC - NxC
        with tf.variable_scope('dense2') as scope:
            last = dense(last, in_channels, use_bias=True, activation='sigmoid',
              initializer=2, init_factor=1.0, collection=collection)
        # scale
        if data_format == 'NCHW':
            last = tf.expand_dims(last, -1)
            last = tf.expand_dims(last, -1)
        else:
            last = tf.expand_dims(last, -2)
            last = tf.expand_dims(last, -2)
        return tf.multiply(skip, last)

def SqueezeExcitationLocal(last, channels=None, channel_r=1, ksize=16,
                           data_format='NHWC', collection=None):
    shape = helper.dim2int(last.get_shape())
    in_channels = shape[-3] if data_format == 'NCHW' else shape[-1]
    if channels is None: channels = in_channels
    channels //= channel_r
    with tf.variable_scope('squeeze_excitation_local') as scope:
        skip = last
        # squeeze - average local features
        with tf.variable_scope('downscale_conv') as scope:
            last = depthwise_conv2d(last, ksize, 1, use_bias=True,
                stride=ksize, padding='SAME', data_format=data_format,
                bn=None, train=False, activation=None,
                initializer=4, init_factor=1.0, collection=collection)
        '''
        if isinstance(ksize, int) or isinstance(ksize, tf.Dimension):
            ksize = [1, 1, ksize, ksize] if data_format == 'NCHW' else [1, ksize, ksize, 1]
        last = tf.nn.avg_pool(last, ksize, strides=ksize,
            padding='SAME', data_format=data_format)
        '''
        # non-linear mapping
        with tf.variable_scope('pointwise_conv1') as scope:
            last = conv2d(last, 1, channels, use_bias=True,
                stride=1, padding='SAME', data_format=data_format,
                bn=None, train=False, activation='swish',
                initializer=4, init_factor=2.0, collection=collection)
        with tf.variable_scope('pointwise_conv2') as scope:
            last = conv2d(last, 1, in_channels, use_bias=True,
                stride=1, padding='SAME', data_format=data_format,
                bn=None, train=False, activation='sigmoid',
                initializer=4, init_factor=2.0, collection=collection)
        # upsample
        if data_format == 'NCHW':
            last = utils.image.NCHW2NHWC(last)
        size = tf.shape(skip)[-2:] if data_format == 'NCHW' else tf.shape(skip)[-3:-1]
        last = tf.image.resize_bilinear(last, size, align_corners=True)
        if data_format == 'NCHW':
            last = utils.image.NHWC2NCHW(last)
        # excitation - feature weighting
        return tf.multiply(skip, last)

def apply_activation(last, activation, data_format='NHWC', collection=None):
    if isinstance(activation, str):
        activation = activation.lower()
    if activation and activation != 'none':
        if activation == 'sigmoid':
            last = tf.sigmoid(last)
        elif activation == 'tanh':
            last = tf.tanh(last)
        elif activation == 'swish':
            with tf.variable_scope('swish') as scope:
                last = last * tf.sigmoid(last)
        elif activation == 'swish_mod1':
            # http://kexue.fm/archives/4647/
            # x * min(1, exp(x))
            with tf.variable_scope('swish_mod1') as scope:
                last = tf.maximum(x, x * tf.exp(tf.negative(tf.abs(x))))
        elif activation == 'relu':
            last = tf.nn.relu(last)
        elif activation == 'prelu':
            last = PReLU(last, data_format, collection)
        elif activation[0:5] == 'lrelu':
            alpha = activation[5:]
            if alpha: alpha = float(alpha)
            else: alpha = 0.3
            lrelu = tf.contrib.keras.layers.LeakyReLU(alpha=alpha)
            last = lrelu(last)
        elif activation == 'elu':
            last = tf.nn.elu(last)
        elif activation == 'crelu':
            last = tf.nn.crelu(last)
        elif activation == 'su':
            last = SelectionUnit(last, data_format, collection)
        elif activation[0:2] == 'se':
            channel_r = activation[2:]
            if channel_r: channel_r = int(channel_r)
            else: channel_r = 1
            last = SqueezeExcitation(last, None, channel_r, data_format, collection)
        else:
            raise ValueError('Unrecognized \'activation\' specified!')
        #activation_summary(last)
    return last

def conv2d(last, ksize, out_channels=None, use_bias=None,
           stride=1, padding='SAME', data_format='NHWC',
           bn=None, train=False, activation=None,
           initializer=4, init_factor=2.0, collection=None):
    # parameters
    in_channels = last.get_shape()[-3 if data_format == 'NCHW' else -1]
    if out_channels is None: out_channels = in_channels
    if isinstance(ksize, int) or isinstance(ksize, tf.Dimension):
        ksize = [ksize, ksize]
    if isinstance(stride, int) or isinstance(stride, tf.Dimension):
        stride = [1, 1, stride, stride] if data_format == 'NCHW' else [1, stride, stride, 1]
    # weight
    kshape = [ksize[0], ksize[1], in_channels, out_channels]
    kernel = conv2d_variable('weights', shape=kshape,
                             initializer=initializer, init_factor=init_factor,
                             collection=collection)
    last = tf.nn.conv2d(last, kernel, strides=stride,
                        padding=padding, data_format=data_format)
    # bias
    if use_bias is None:
        use_bias = not bn
    if use_bias:
        biases = get_variable('biases', [out_channels], tf.zeros_initializer())
        last = tf.nn.bias_add(last, biases, data_format=data_format)
    # batch normalization
    last = apply_batch_norm(last, decay=bn, train=train,
        data_format=data_format)
    # activation function
    last = apply_activation(last, activation, data_format, collection)
    return last

def depthwise_conv2d(last, ksize, channel_multiplier=1, use_bias=None,
                     stride=1, padding='SAME', data_format='NHWC',
                     bn=None, train=False, activation=None,
                     initializer=4, init_factor=2.0, collection=None):
    # parameters
    in_channels = last.get_shape()[-3 if data_format == 'NCHW' else -1]
    out_channels = in_channels * channel_multiplier
    if isinstance(ksize, int) or isinstance(ksize, tf.Dimension):
        ksize = [ksize, ksize]
    if isinstance(stride, int) or isinstance(stride, tf.Dimension):
        stride = [1, 1, stride, stride] if data_format == 'NCHW' else [1, stride, stride, 1]
    # weight
    depthwise_kshape = [ksize[0], ksize[1], in_channels, channel_multiplier]
    depthwise_kernel = conv2d_variable('depthwise_weights', shape=depthwise_kshape,
                                       initializer=initializer, init_factor=init_factor,
                                       collection=collection)
    last = tf.nn.depthwise_conv2d_native(last, depthwise_kernel, strides=stride,
                                         padding=padding, data_format=data_format)
    # bias
    if use_bias is None:
        use_bias = not bn
    if use_bias:
        biases = get_variable('biases', [out_channels], tf.zeros_initializer())
        last = tf.nn.bias_add(last, biases, data_format=data_format)
    # batch normalization
    last = apply_batch_norm(last, decay=bn, train=train,
        data_format=data_format)
    # activation function
    last = apply_activation(last, activation, data_format, collection)
    return last

def separable_conv2d(last, ksize, channel_multiplier=1, out_channels=None, use_bias=None,
                     stride=1, padding='SAME', data_format='NHWC',
                     bn=None, train=False, activation=None,
                     initializer=4, init_factor=2.0, collection=None):
    # parameters
    in_channels = last.get_shape()[-3 if data_format == 'NCHW' else -1]
    temp_channels = in_channels * channel_multiplier
    if out_channels is None: out_channels = temp_channels
    if isinstance(ksize, int) or isinstance(ksize, tf.Dimension):
        ksize = [ksize, ksize]
    if isinstance(stride, int) or isinstance(stride, tf.Dimension):
        stride = [1, 1, stride, stride] if data_format == 'NCHW' else [1, stride, stride, 1]
    # weight
    depthwise_kshape = [ksize[0], ksize[1], in_channels, channel_multiplier]
    pointwise_kshape = [1, 1, temp_channels, out_channels]
    depthwise_kernel = conv2d_variable('depthwise_weights', shape=depthwise_kshape,
                                       initializer=initializer, init_factor=init_factor,
                                       collection=collection)
    pointwise_kernel = conv2d_variable('pointwise_weights', shape=pointwise_kshape,
                                       initializer=initializer, init_factor=init_factor,
                                       collection=collection)
    last = tf.nn.separable_conv2d(last, depthwise_kernel, pointwise_kernel, strides=stride,
                                  padding=padding, data_format=data_format)
    # bias
    if use_bias is None:
        use_bias = not bn
    if use_bias:
        biases = get_variable('biases', [out_channels], tf.zeros_initializer())
        last = tf.nn.bias_add(last, biases, data_format=data_format)
    # batch normalization
    last = apply_batch_norm(last, decay=bn, train=train,
        data_format=data_format)
    # activation function
    last = apply_activation(last, activation, data_format, collection)
    return last

def deconv2d(last, ksize, out_channels=None, out_size=None, use_bias=None,
             stride=1, padding='SAME', data_format='NHWC',
             bn=None, train=False, activation=None,
             initializer=4, init_factor=2.0, collection=None):
    # parameters
    in_channels = last.get_shape()[-3 if data_format == 'NCHW' else -1]
    if out_channels is None: out_channels = in_channels
    if isinstance(ksize, int) or isinstance(ksize, tf.Dimension):
        ksize = [ksize, ksize]
    shape = tf.shape(last)
    if data_format == 'NCHW':
        if out_size is None: out_size = [shape[2], shape[3]]
        out_shape = [shape[0], out_channels, out_size[0], out_size[1]]
    else:
        if out_size is None: out_size = [shape[1], shape[2]]
        out_shape = [shape[0], out_size[0], out_size[1], out_channels]
    if isinstance(stride, int) or isinstance(stride, tf.Dimension):
        stride = [1, 1, stride, stride] if data_format == 'NCHW' else [1, stride, stride, 1]
    # weight
    kshape = [ksize[0], ksize[1], out_channels, in_channels]
    kernel = conv2d_variable('weights', shape=kshape,
                             initializer=initializer, init_factor=init_factor,
                             collection=collection)
    last = tf.nn.conv2d_transpose(last, kernel, out_shape, strides=stride,
                                  padding=padding, data_format=data_format)
    # bias
    if use_bias is None:
        use_bias = not bn
    if use_bias:
        biases = get_variable('biases', [out_channels], tf.zeros_initializer())
        last = tf.nn.bias_add(last, biases, data_format=data_format)
    # batch normalization
    last = apply_batch_norm(last, decay=bn, train=train,
        data_format=data_format)
    # activation function
    last = apply_activation(last, activation, data_format, collection)
    return last

# checkerboard artifacts free resize convolution
# https://distill.pub/2016/deconv-checkerboard/
def resize_conv2d(last, ksize, out_channels=None, use_bias=None,
                  scaling=2, padding='SAME', data_format='NHWC',
                  bn=None, train=False, activation=None,
                  initializer=4, init_factor=2.0, collection=None):
    # parameters
    if isinstance(scaling, int) or isinstance(scaling, tf.Dimension):
        scaling = [scaling, scaling]
    shape = tf.shape(last)
    if data_format == 'NCHW':
        out_size = [shape[2] * scaling[0], shape[3] * scaling[1]]
    else:
        out_size = [shape[1] * scaling[0], shape[2] * scaling[1]]
    # nearest-neighbor upsample
    if data_format == 'NCHW':
        last = utils.image.NCHW2NHWC(last)
    last = tf.image.resize_nearest_neighbor(last, size=out_size)
    if data_format == 'NCHW':
        last = utils.image.NHWC2NCHW(last)
    # convolution 2D
    last = conv2d(last, ksize, out_channels=out_channels, use_bias=use_bias,
        stride=1, padding=padding, data_format=data_format,
        bn=bn, train=train, activation=activation,
        initializer=initializer, init_factor=init_factor, collection=collection)
    return last

def depthwise_resize_conv2d(last, ksize, channel_multiplier=1, use_bias=None,
                            scaling=2, padding='SAME', data_format='NHWC',
                            bn=None, train=False, activation=None,
                            initializer=4, init_factor=2.0, collection=None):
    # parameters
    if isinstance(scaling, int) or isinstance(scaling, tf.Dimension):
        scaling = [scaling, scaling]
    shape = tf.shape(last)
    if data_format == 'NCHW':
        out_size = [shape[2] * scaling[0], shape[3] * scaling[1]]
    else:
        out_size = [shape[1] * scaling[0], shape[2] * scaling[1]]
    # nearest-neighbor upsample
    if data_format == 'NCHW':
        last = utils.image.NCHW2NHWC(last)
    last = tf.image.resize_nearest_neighbor(last, size=out_size)
    if data_format == 'NCHW':
        last = utils.image.NHWC2NCHW(last)
    # convolution 2D
    last = depthwise_conv2d(last, ksize, channel_multiplier=channel_multiplier, use_bias=use_bias,
        stride=1, padding=padding, data_format=data_format,
        bn=bn, train=train, activation=activation,
        initializer=initializer, init_factor=init_factor, collection=collection)
    return last

def subpixel_conv2d(last, ksize, out_channels=None, use_bias=None,
                    scaling=2, padding='SAME', data_format='NHWC',
                    bn=None, train=False, activation=None,
                    initializer=4, init_factor=2.0, collection=None):
    # parameters
    if isinstance(ksize, int) or isinstance(ksize, tf.Dimension):
        ksize = [ksize, ksize]
    if isinstance(scaling, int) or isinstance(scaling, tf.Dimension):
        scaling = [scaling, scaling]
    for _ in range(2):
        if ksize[_] % scaling[_] != 0:
            raise ValueError('subpixel_conv2d: \'ksize\' must be divisible by \'scaling\'!')
    shape = tf.shape(last)
    if data_format == 'NCHW':
        strides = [1, 1, scaling[0], scaling[1]]
    else:
        strides = [1, scaling[0], scaling[1], 1]
    # parameters
    in_channels = last.get_shape()[-3 if data_format == 'NCHW' else -1]
    if out_channels is None: out_channels = in_channels
    if isinstance(scaling, int) or isinstance(scaling, tf.Dimension):
        scaling = [scaling, scaling]
    temp_channels = out_channels * scaling[0] * scaling[1]
    # convolution 2D
    last = conv2d(last, ksize, out_channels=temp_channels, use_bias=use_bias,
        stride=1, padding=padding, data_format=data_format,
        bn=bn, train=train, activation=activation,
        initializer=initializer, init_factor=init_factor, collection=collection)
    # periodic shuffling
    last = tf.depth_to_space(last, scaling[0], data_format=data_format)
    return last
