import sys
import re
import numpy as np
import tensorflow as tf
sys.path.append('..')
import utils.image

# flags
FLAGS = tf.app.flags.FLAGS

# basic parameters
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")
tf.app.flags.DEFINE_boolean('multiGPU', False,
                            """Train the model using multiple GPUs.""")
tf.app.flags.DEFINE_integer('scaling', 2,
                            """Scaling ratio of the super-resolution filter.""")
tf.app.flags.DEFINE_float('weight_decay', 0, #1e-4,
                            """L2 regularization weight decay factor""")
tf.app.flags.DEFINE_float('learning_rate', 1e-3,
                            """Initial learning rate""")
tf.app.flags.DEFINE_float('lr_min', 1e-4,
                            """Minimum learning rate""")
tf.app.flags.DEFINE_float('lr_decay_steps', 1e3,
                            """Steps after which learning rate decays""")
tf.app.flags.DEFINE_float('lr_decay_factor', 0.95,
                            """Learning rate decay factor""")
tf.app.flags.DEFINE_float('learning_momentum', 0.9,
                            """momentum for MomentumOptimizer""")
tf.app.flags.DEFINE_float('learning_beta1', 0.9,
                            """beta1 for AdamOptimizer""")
tf.app.flags.DEFINE_float('epsilon', 1e-8,
                            """Fuzz term to avoid numerical instability""")
tf.app.flags.DEFINE_float('gradient_clipping', 0, #0.002,
                            """Gradient clipping factor""")
tf.app.flags.DEFINE_float('loss_moving_average', 0.9,
                            """The decay to use for the moving average of losses""")
tf.app.flags.DEFINE_float('train_moving_average', 0.9999,
                            """The decay to use for the moving average of trainable variables""")

# advanced model parameters
tf.app.flags.DEFINE_integer('k_first', 3,
                            """Kernel size for the first layer.""")
tf.app.flags.DEFINE_integer('k_last', 3,
                            """Kernel size for the last layer.""")
tf.app.flags.DEFINE_integer('res_blocks', 6,
                            """Number of residual blocks.""")
tf.app.flags.DEFINE_integer('channels', 64,
                            """Number of features in hidden layers.""")
tf.app.flags.DEFINE_integer('channels2', 32,
                            """Number of features after resize conv.""")
tf.app.flags.DEFINE_float('batch_norm', 0, #0.999,
                            """Moving average decay for Batch Normalization.""")
tf.app.flags.DEFINE_string('activation', 'relu',
                            """Activation function used.""")
tf.app.flags.DEFINE_integer('initializer', 5,
                            """Weights initialization method.""")
tf.app.flags.DEFINE_float('init_factor', 1.0,
                            """Weights initialization STD factor for conv layers without activation.""")
tf.app.flags.DEFINE_float('init_activation', 1.0,
                            """Weights initialization STD factor for conv layers with activation.""")

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

# functions
def _activation_summary(x):
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

def _get_variable(name, shape, initializer, trainable=True):
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

def _conv2d_variable(name, shape, init_factor=None, wd=None):
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
    shape = [int(s) for s in shape]
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
        stddev = np.sqrt(init_factor / (np.sqrt(shape[2] * shape[3]) * shape[0] * shape[1]))
        initializer = tf.truncated_normal_initializer(stddev=stddev, dtype=dtype)
    # weights initialization
    var = _get_variable(name, shape, initializer)
    # L2 regularization (weight decay)
    if wd is not None and wd != 0:
        regularizer = tf.contrib.layers.l2_regularizer(wd)
        regularizer(var)
    return var

def _conv2d(scope, last, ksize, out_channels, stride=1, padding='SAME',
            batch_norm=0.999, is_training=True, activation='relu',
            init_factor=1.0, wd=None):
    # parameters
    shape = last.get_shape()
    in_channels = shape[-1]
    kshape = [ksize, ksize, in_channels, out_channels]
    kernel = _conv2d_variable('weights', shape=kshape,
                              init_factor=init_factor, wd=wd)
    # convolution 2D
    last = tf.nn.conv2d(last, kernel, [1, 1, 1, 1], padding=padding)
    biases = _get_variable('biases', [out_channels],
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
        _activation_summary(last)
    return last

def _resize_conv2d(scope, last, ksize, out_channels, scaling=2,
                   batch_norm=None, is_training=True, activation=None,
                   init_factor=1.0, wd=None):
    # parameters
    shape = last.get_shape()
    in_channels = shape[-1]
    kshape = [ksize, ksize, out_channels, in_channels]
    out_shape = [shape[0], shape[1] * scaling, shape[2] * scaling, out_channels]
    out_shape = [int(d) for d in out_shape]
    # nearest-neighbor upsample
    last = tf.image.resize_nearest_neighbor(last, size=out_shape[1:3])
    # deconvolution 2D
    kernel = _conv2d_variable('weights', shape=kshape,
                              init_factor=init_factor, wd=wd)
    last = tf.nn.conv2d_transpose(last, kernel, out_shape,
                                  [1, 1, 1, 1], padding='SAME')
    biases = _get_variable('biases', [out_channels],
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
        _activation_summary(last)
    return last

# model
def inference(images_lr, is_training):
    print('k_first={}, k_last={}, res_blocks={}, channels={}, channels2={}'.format(
        FLAGS.k_first, FLAGS.k_last, FLAGS.res_blocks, FLAGS.channels, FLAGS.channels2))
    last = images_lr
    l = 0
    # channels
    image_channels = images_lr.get_shape()[-1]
    conv_layers = 1 + FLAGS.res_blocks * 2 + 1
    conv2_layers = 1
    channels = [image_channels] + \
        [FLAGS.channels for l in range(conv_layers)] + \
        [FLAGS.channels2 for l in range(conv2_layers)]
    # first conv layer
    l += 1
    with tf.variable_scope('conv{}'.format(l)) as scope:
        last = _conv2d(scope, last, ksize=FLAGS.k_first, out_channels=channels[l],
                       stride=1, padding='SAME',
                       batch_norm=None, is_training=is_training, activation=FLAGS.activation,
                       init_factor=FLAGS.init_activation, wd=FLAGS.weight_decay)
    skip1 = last
    # residual blocks
    rb = 0
    while rb < FLAGS.res_blocks:
        rb += 1
        skip2 = last 
        l += 1
        with tf.variable_scope('conv{}'.format(l)) as scope:
            last = _conv2d(scope, last, ksize=3, out_channels=channels[l],
                           stride=1, padding='SAME',
                           batch_norm=FLAGS.batch_norm, is_training=is_training, activation=FLAGS.activation,
                           init_factor=FLAGS.init_activation, wd=FLAGS.weight_decay)
        l += 1
        with tf.variable_scope('conv{}'.format(l)) as scope:
            last = _conv2d(scope, last, ksize=3, out_channels=channels[l],
                           stride=1, padding='SAME',
                           batch_norm=FLAGS.batch_norm, is_training=is_training, activation=None,
                           init_factor=FLAGS.init_factor, wd=FLAGS.weight_decay)
            last = tf.add(last, skip2, 'elementwise_sum')
    # skip connection
    l += 1
    with tf.variable_scope('conv{}'.format(l)) as scope:
        last = _conv2d(scope, last, ksize=3, out_channels=channels[l],
                       stride=1, padding='SAME',
                       batch_norm=FLAGS.batch_norm, is_training=is_training, activation=None,
                       init_factor=FLAGS.init_factor, wd=FLAGS.weight_decay)
        last = tf.add(last, skip1, 'elementwise_sum')
    # resize conv layer
    l += 1
    with tf.variable_scope('resize_conv{}'.format(l)) as scope:
        last = _resize_conv2d(scope, last, ksize=3, out_channels=channels[l], scaling=FLAGS.scaling,
                              batch_norm=None, is_training=is_training, activation=FLAGS.activation,
                              init_factor=FLAGS.init_activation, wd=FLAGS.weight_decay)
    # final conv layer
    l += 1
    with tf.variable_scope('conv{}'.format(l)) as scope:
        last = _conv2d(scope, last, ksize=FLAGS.k_last, out_channels=image_channels,
                       stride=1, padding='SAME',
                       batch_norm=None, is_training=is_training, activation=None,
                       init_factor=FLAGS.init_factor, wd=FLAGS.weight_decay)
    # return SR image
    print('Totally {} convolutional layers.'.format(l))
    return last

def main_loss(images_hr, images_sr):
    # RGB loss
    RGB_mad = tf.losses.absolute_difference(images_hr, images_sr, weights=1e-3)
    # OPP loss
    images_hr = utils.image.RGB2OPP(images_hr, norm=False)
    images_sr = utils.image.RGB2OPP(images_sr, norm=False)
    #mse = tf.losses.mean_squared_error(images_hr, images_sr, weights=1.0)
    mad = tf.losses.absolute_difference(images_hr, images_sr, weights=1.0)
    return mad, RGB_mad

def loss():
    return tf.losses.get_total_loss()

def _add_loss_summaries(losses, decay):
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

def train(total_loss, global_step, epoch_size):
    # decay the learning rate exponentially based on the number of steps
    lr = FLAGS.learning_rate
    if FLAGS.lr_decay_steps > 0 and FLAGS.lr_decay_factor != 1:
        lr = tf.train.exponential_decay(lr, global_step,
                                        FLAGS.lr_decay_steps, FLAGS.lr_decay_factor,
                                        staircase=True)
        lr = tf.maximum(FLAGS.lr_min, lr)
    tf.summary.scalar('learning_rate', lr)
    
    # dependency need to be updated
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    
    # generate moving averages of all losses and associated summaries
    losses = tf.losses.get_losses()
    loss_averages_op = _add_loss_summaries(losses, FLAGS.loss_moving_average)
    if loss_averages_op: update_ops.append(loss_averages_op)

    # compute gradients
    with tf.control_dependencies(update_ops):
        #opt = tf.train.MomentumOptimizer(lr, momentum=FLAGS.learning_momentum, use_nesterov=True)
        opt = tf.train.AdamOptimizer(lr, beta1=FLAGS.learning_beta1, epsilon=FLAGS.epsilon)
        grads_and_vars = opt.compute_gradients(total_loss)

    # gradient clipping
    if FLAGS.gradient_clipping > 0:
        clip_value = FLAGS.gradient_clipping / lr
        grads_and_vars = [(tf.clip_by_value(
                grad, -clip_value, clip_value, name='gradient_clipping'
                ), var) for grad, var in grads_and_vars]

    # training ops
    train_ops = []
    
    # apply gradient
    apply_gradient_op = opt.apply_gradients(grads_and_vars, global_step)
    train_ops.append(apply_gradient_op)

    # add histograms for trainable variables
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # add histograms for gradients
    for grad, var in grads_and_vars:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    # track the moving averages of all trainable variables
    if FLAGS.train_moving_average:
        variable_averages = tf.train.ExponentialMovingAverage(
                FLAGS.train_moving_average, global_step, name='train_moving_average')
        variable_averages_op = variable_averages.apply(tf.trainable_variables())
        train_ops.append(variable_averages_op)
    
    # generate operation
    with tf.control_dependencies(train_ops):
        train_op = tf.no_op(name='train')
    return train_op
