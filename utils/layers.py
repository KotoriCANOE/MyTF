import re
import numpy as np
import tensorflow as tf
from utils import helper

# flags
FLAGS = tf.app.flags.FLAGS

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

def loss_summaries(losses, decay):
    if decay:
        loss_averages = tf.train.ExponentialMovingAverage(decay, name='loss_moving_average')
        loss_averages_op = loss_averages.apply(losses)
    else:
        loss_averages_op = None
    
    for l in losses:
        tf.summary.scalar(l.op.name + '.raw', l)
        if loss_averages_op:
            tf.summary.scalar(l.op.name, loss_averages.average(l))
    return loss_averages_op

def get_variable(name, shape, initializer, trainable=True):
    """Helper to create a Variable stored on CPU memory.
    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable
    Returns:
      Variable Tensor
    """
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, dtype=dtype,
                              initializer=initializer,
                              trainable=trainable)
    return var

def conv2d_variable(name, shape, init_factor=None, wd=None):
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
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    shape = helper.dim2int(shape)
    batch_size = FLAGS.batch_size
    # weights initializer
    if init_factor is None:
        init_factor = 1.0 if FLAGS.initializer == 4 else 2.0
    if FLAGS.initializer == 1: # uniform Xavier initializer
        initializer = tf.contrib.layers.variance_scaling_initializer(
            factor=init_factor, mode='FAN_AVG', uniform=True)
    elif FLAGS.initializer == 2: # normal Xavier initializer
        initializer = tf.contrib.layers.variance_scaling_initializer(
            factor=init_factor, mode='FAN_AVG', uniform=False)
    elif FLAGS.initializer == 3: # Convolutional Architecture for Fast Feature Embedding
        initializer = tf.contrib.layers.variance_scaling_initializer(
            factor=init_factor, mode='FAN_IN', uniform=True)
    elif FLAGS.initializer == 4: # Delving Deep into Rectifiers, init_factor should be 2.0 for ReLU
        initializer = tf.contrib.layers.variance_scaling_initializer(
            factor=init_factor, mode='FAN_IN', uniform=False)
    elif FLAGS.initializer >= 5: # modified Xavier initializer
        stddev = np.sqrt(init_factor / (np.sqrt(shape[2] * shape[3]) * batch_size * shape[1]))
        initializer = tf.truncated_normal_initializer(stddev=stddev, dtype=dtype)
    # weights initialization
    var = get_variable(name, shape, initializer)
    # L2 regularization (weight decay)
    if wd is not None and wd != 0:
        regularizer = tf.contrib.layers.l2_regularizer(wd)
        regularizer(var)
    return var

def conv2d(scope, last, ksize, out_channels, stride=1, padding='SAME',
            batch_norm=0.999, is_training=True, activation='relu',
            init_factor=1.0, wd=None):
    # parameters
    shape = last.get_shape()
    in_channels = shape[-1]
    kshape = [ksize, ksize, in_channels, out_channels]
    kernel = conv2d_variable('weights', shape=kshape,
                              init_factor=init_factor, wd=wd)
    # convolution 2D
    last = tf.nn.conv2d(last, kernel, [1, 1, 1, 1], padding=padding)
    biases = get_variable('biases', [out_channels],
                           tf.constant_initializer(0.0))
    last = tf.nn.bias_add(last, biases)
    # batch normalization
    if batch_norm:
        last = tf.contrib.layers.batch_norm(last, decay=batch_norm,
                                            is_training=is_training)
    # activation function
    if isinstance(activation, str):
        activation = activation.lower()
    if activation and activation != 'none':
        if activation == 'relu':
            last = tf.nn.relu(last, name=scope.name)
        elif activation == 'prelu':
            prelu = tf.contrib.keras.layers.PReLU(shared_axes=[1, 2], name=scope.name)
            last = prelu(last)
        elif activation[0:5] == 'lrelu':
            alpha = activation[5:]
            if alpha: alpha = float(alpha)
            else: alpha = 0.3
            lrelu = tf.contrib.keras.layers.LeakyReLU(alpha=alpha, name=scope.name)
            last = lrelu(last)
        else:
            raise ValueError('Unrecognized \'activation\' specified!')
        activation_summary(last)
    return last

def resize_conv2d(scope, last, ksize, out_channels, scaling=2,
                   batch_norm=None, is_training=True, activation=None,
                   init_factor=1.0, wd=None):
    # parameters
    shape = last.get_shape()
    in_channels = shape[-1]
    kshape = [ksize, ksize, out_channels, in_channels]
    out_shape = [shape[0], shape[1] * scaling, shape[2] * scaling, out_channels]
    # nearest-neighbor upsample
    last = tf.image.resize_nearest_neighbor(last, size=helper.dim2int(out_shape)[1:3])
    # deconvolution 2D
    kernel = conv2d_variable('weights', shape=kshape,
                              init_factor=init_factor, wd=wd)
    last = tf.nn.conv2d_transpose(last, kernel, tf.TensorShape(out_shape),
                                  [1, 1, 1, 1], padding='SAME')
    biases = get_variable('biases', [out_channels],
                              tf.constant_initializer(0.0))
    last = tf.nn.bias_add(last, biases)
    # batch normalization
    if batch_norm:
        last = tf.contrib.layers.batch_norm(last, decay=batch_norm,
                                            is_training=is_training)
    # activation function
    if isinstance(activation, str):
        activation = activation.lower()
    if activation and activation != 'none':
        if activation == 'relu':
            last = tf.nn.relu(last, name=scope.name)
        elif activation == 'prelu':
            prelu = tf.contrib.keras.layers.PReLU(shared_axes=[1, 2], name=scope.name)
            last = prelu(last)
        elif activation[0:5] == 'lrelu':
            alpha = activation[5:]
            if alpha: alpha = float(alpha)
            else: alpha = 0.3
            lrelu = tf.contrib.keras.layers.LeakyReLU(alpha=alpha, name=scope.name)
            last = lrelu(last)
        else:
            raise ValueError('Unrecognized \'activation\' specified!')
        activation_summary(last)
    return last
