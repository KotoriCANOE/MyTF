import re
import tensorflow as tf

# flags
FLAGS = tf.app.flags.FLAGS

# basic model parameters
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")
tf.app.flags.DEFINE_boolean('multiGPU', False,
                            """Train the model using multiple GPUs.""")
tf.app.flags.DEFINE_integer('scaling', 2,
                            """Scaling ratio of the super-resolution filter.""")

# advanced model parameters
CONV_LAYERS = 9
CHANNELS = 64

# constants describing the training process
WEIGHT_DECAY = 0.0001
MOVING_AVERAGE_DECAY = 0.9999 # the decay to use for the moving average
EPOCHS_PER_DECAY = 20.0 # epochs after which learning rate decays
INITIAL_LEARNING_RATE = 0.1 # initial learning rate
LEARNING_RATE_DECAY_FACTOR = 0.55 # learning rate decay factor
GRADIENT_CLIPPING = 0.0005 # gradient clipping factor

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

def _get_variable(name, shape, initializer):
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
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var

def _variable_with_weight_decay(name, shape, stddev, wd):
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
    var = _get_variable(name, shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None and wd != 0:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

# model
def inference(images_lr, images_up, channels=CHANNELS):
    scaling = FLAGS.scaling
    image_channels = images_lr.shape[-1]
    if isinstance(channels, int):
        channels = [image_channels] + [channels for l in range(CONV_LAYERS)]
    last = images_lr
    l = 0
    # conv layers
    while l < CONV_LAYERS:
        l += 1
        with tf.variable_scope('conv{}'.format(l)) as scope:
            in_channels = last.shape[-1]
            out_channels = channels[l]
            kshape = [3, 3, in_channels, out_channels]
            kernel = _variable_with_weight_decay('weights', shape=kshape,
                                                 stddev=0.001, wd=WEIGHT_DECAY)
            conv = tf.nn.conv2d(last, kernel, [1, 1, 1, 1], padding='VALID')
            biases = _get_variable('biases', [out_channels],
                                      tf.constant_initializer(0.0))
            biased = tf.nn.bias_add(conv, biases)
            activated = tf.nn.relu(biased, name=scope.name)
            #_activation_summary(activated)
            last = activated
    # final deconv layer
    l += 1
    with tf.variable_scope('deconv{}'.format(l)) as scope:
        in_channels = last.shape[-1]
        out_channels = image_channels
        kshape = [5, 5, out_channels, in_channels]
        out_shape = [last.shape[0], last.shape[1] * scaling, last.shape[1] * scaling, out_channels]
        out_shape = [int(d) for d in out_shape]
        print('Deconvolution input shape: {}'.format(last.get_shape()))
        kernel = _variable_with_weight_decay('weights', shape=kshape,
                                             stddev=0.001, wd=WEIGHT_DECAY)
        conv = tf.nn.conv2d_transpose(last, kernel, out_shape,
                                      [1, scaling, scaling, 1], padding='SAME')
        print('Deconvolution output shape: {}'.format(conv.get_shape()))
        biases = _get_variable('biases', [out_channels],
                                  tf.constant_initializer(0.0))
        biased = tf.nn.bias_add(conv, biases)
        #activated = tf.nn.relu(biased, name=scope.name)
        #_activation_summary(activated)
        last = biased
    # get SR image by adding residual to upsampled LR image
    with tf.variable_scope('residual_add') as scope:
        residual = last
        #crop_w = CONV_LAYERS
        #crop_h = CONV_LAYERS
        #images_lr_crop = images_lr[:, crop_h:-crop_h, crop_w:-crop_w, :]
        #size_up = [int(residual.shape[1]), int(residual.shape[2])]
        #images_up = tf.image.resize_nearest_neighbor(images_lr_crop, size_up)
        images_sr = tf.add(images_up, residual, name='add')
    # return SR image
    return images_sr

def main_loss(images_sr, images_hr):
    #crop_w = CONV_LAYERS * FLAGS.scaling
    #crop_h = CONV_LAYERS * FLAGS.scaling
    #images_sr = images_sr[:, crop_h:-crop_h, crop_w:-crop_w, :]
    #images_hr = images_hr[:, crop_h:-crop_h, crop_w:-crop_w, :]
    #print('Cropped HR shape: {}'.format(images_hr.get_shape()))
    '''
    squared_error = tf.squared_difference(images_sr, images_hr, name='squared_error')
    mean_squared_error = tf.reduce_mean(squared_error, name='mean_squared_error')
    tf.add_to_collection('losses', mean_squared_error)
    return mean_squared_error
    '''
    abs_diff = tf.abs(tf.subtract(images_sr, images_hr, name='subtract'),
                      name='absolute_difference')
    mean_abs_diff = tf.reduce_mean(abs_diff, name='mean_absolute_difference')
    tf.add_to_collection('losses', mean_abs_diff)
    return mean_abs_diff

def loss():
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def _add_loss_summaries(total_loss):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    
    for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name + '.raw', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))
    return loss_averages_op

def train(total_loss, global_step, epoch_size):
    # variables that affect learning rate
    batches_per_epoch = epoch_size / FLAGS.batch_size
    decay_steps = int(batches_per_epoch * EPOCHS_PER_DECAY)
    
    # decay the learning rate exponentially based on the number of steps
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step,
                                    decay_steps, LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)
    
    # generate moving averages of all losses and associated summaries
    loss_averages_op = _add_loss_summaries(total_loss)

    # compute gradients
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads_and_vars = opt.compute_gradients(total_loss)

    # gradient clipping
    clip_value = GRADIENT_CLIPPING / lr
    grads_and_vars = [(tf.clip_by_value(
            grad, -clip_value, clip_value, name='gradient_clipping'
            ), var) for grad, var in grads_and_vars]

    # apply gradient
    apply_gradient_op = opt.apply_gradients(grads_and_vars, global_step)

    # add histograms for trainable variables
    #for var in tf.trainable_variables():
    #    tf.summary.histogram(var.op.name, var)

    # add histograms for gradients
    #for grad, var in grads:
    #    if grad is not None:
    #        tf.summary.histogram(var.op.name + '/gradients', grad)

    # track the moving averages of all trainable variables
    #variable_averages = tf.train.ExponentialMovingAverage(
    #        MOVING_AVERAGE_DECAY, global_step)
    #variable_averages_op = variable_averages.apply(tf.trainable_variables())
    
    # generate operation
    #with tf.control_dependencies([apply_gradient_op, variable_averages_op]):
    with tf.control_dependencies([apply_gradient_op]):
        train_op = tf.no_op(name='train')
    return train_op

'''
def VDSR(epochs, batch_size, learn_rate=0.1, weight_decay=0.0001, layers=9, channels=64, image_channels=3):
    if isinstance(channels, int): channels = [channels for l in range(layers)]
    shape = (None, None, None, image_channels)
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32

    x = tf.placeholder(dtype, shape, name='LR_image')
    x2 = tf.placeholder(dtype, shape, name='LR_image_upsampled')
    y_ = tf.placeholder(dtype, shape, name='HR_image')
    weights = []
    biases = []

    # conv layers
    last = x
    for l in range(layers):
        last, W, b = conv2dLayer(last, channels[l], 3, 1, padding='SAME', activation='relu',
                                 weight_std=1e-3, bias=0, wd=weight_decay)
        weights.append(W)
        biases.append(b)

    # final deconv layer
    last, W, b = deconv2dLayer(last, image_channels, 3, 2, padding='SAME', activation='',
                               weight_std=1e-3, bias=0, wd=weight_decay)
    weights.append(W)
    biases.append(b)

    # adding residual
    y = x2 + last

    # loss function
    l2loss = tf.reduce_mean(tf.squared_difference(y, y_, name='loss (L2 diff)'), name='loss (mean)')
    tf.add_to_collection('losses', l2loss)
    loss = tf.add_n(tf.get_collection('losses'), name='total loss')

    # optimization
    train_step = tf.train.AdamOptimizer(learn_rate, epsilon=1e-3).minimize(loss)

    return train_step, l2loss, weights, biases
'''
