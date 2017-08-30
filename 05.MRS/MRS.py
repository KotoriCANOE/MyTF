import sys
import tensorflow as tf
sys.path.append('..')
from utils import layers

# flags
FLAGS = tf.app.flags.FLAGS

# basic parameters
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")
tf.app.flags.DEFINE_boolean('multiGPU', False,
                            """Train the model using multiple GPUs.""")
tf.app.flags.DEFINE_float('weight_decay', 1e-5,
                            """L2 regularization weight decay factor""")
tf.app.flags.DEFINE_float('learning_rate', 1e-4,
                            """Initial learning rate""")
tf.app.flags.DEFINE_float('lr_min', 0.0,
                            """Minimum learning rate""")
tf.app.flags.DEFINE_float('lr_decay_steps', 1e3,
                            """Steps after which learning rate decays""")
tf.app.flags.DEFINE_float('lr_decay_factor', 0.98,
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
tf.app.flags.DEFINE_integer('k_first', 7,
                            """Kernel size for the first layer.""")
tf.app.flags.DEFINE_integer('res_blocks', 8,
                            """Number of residual blocks.""")
tf.app.flags.DEFINE_integer('channels', 48,
                            """Number of features in hidden layers.""")
tf.app.flags.DEFINE_float('batch_norm', 0, #0.999,
                            """Moving average decay for Batch Normalization.""")
tf.app.flags.DEFINE_string('activation', 'relu',
                            """Activation function used.""")
tf.app.flags.DEFINE_integer('initializer', 3,
                            """Weights initialization method.""")
tf.app.flags.DEFINE_float('init_factor', 1.0,
                            """Weights initialization STD factor for conv layers without activation.""")
tf.app.flags.DEFINE_float('init_activation', 1.0,
                            """Weights initialization STD factor for conv layers with activation.""")

# model
def inference(spectrum, is_training):
    print('k_first={}, res_blocks={}, channels={}'.format(
        FLAGS.k_first, FLAGS.res_blocks, FLAGS.channels))
    weight_decay = FLAGS.weight_decay if is_training else None
    channel_index = -3 if FLAGS.data_format == 'NCHW' else -1
    pool_ksize = [1, 1, 1, 3] if FLAGS.data_format == 'NCHW' else [1, 1, 3, 1]
    reduce_strides = [1, 1, 1, 2] if FLAGS.data_format == 'NCHW' else [1, 1, 2, 1]
    # channels
    channels = FLAGS.channels
    # initialization
    last = spectrum
    l = 0
    # first conv layer
    l += 1
    with tf.variable_scope('conv{}'.format(l)) as scope:
        last = layers.conv2d(last, ksize=[1, FLAGS.k_first], out_channels=channels,
                             stride=[1, 1, 1, 1], padding='SAME', data_format=FLAGS.data_format,
                             batch_norm=None, is_training=is_training, activation=FLAGS.activation,
                             init_factor=FLAGS.init_activation, wd=weight_decay)
    # residual blocks
    rb = 0
    skip2 = last
    while rb < FLAGS.res_blocks:
        rb += 1
        reduce_size = rb % 1 == 0
        strides = reduce_strides if reduce_size else [1, 1, 1, 1]
        double_channel = rb % 3 == 1
        if double_channel: channels *= 2
        l += 1
        with tf.variable_scope('conv{}'.format(l)) as scope:
            last = layers.conv2d(last, ksize=[1, 1], out_channels=channels // 2,
                                 stride=1, padding='SAME', data_format=FLAGS.data_format,
                                 batch_norm=FLAGS.batch_norm, is_training=is_training, activation=FLAGS.activation,
                                 init_factor=FLAGS.init_activation, wd=weight_decay)
        l += 1
        with tf.variable_scope('conv{}'.format(l)) as scope:
            last = layers.conv2d(last, ksize=[1, 3], out_channels=channels // 2,
                                 stride=strides, padding='SAME', data_format=FLAGS.data_format,
                                 batch_norm=FLAGS.batch_norm, is_training=is_training, activation=FLAGS.activation,
                                 init_factor=FLAGS.init_activation, wd=weight_decay)
        l += 1
        with tf.variable_scope('conv{}'.format(l)) as scope:
            last = layers.conv2d(last, ksize=[1, 1], out_channels=channels,
                                 stride=1, padding='SAME', data_format=FLAGS.data_format,
                                 batch_norm=FLAGS.batch_norm, is_training=is_training, activation=None,
                                 init_factor=FLAGS.init_factor, wd=weight_decay)
        with tf.variable_scope('skip_connection{}'.format(l)) as scope:
            if reduce_size:
                skip2 = tf.nn.avg_pool(skip2, ksize=reduce_strides, strides=reduce_strides,
                                       padding='SAME', data_format=FLAGS.data_format)
            if double_channel:
                padding = [[0, 0] for _ in range(4)]
                padding[channel_index] = [0, channels // 2]
                skip2 = tf.pad(skip2, padding, mode='CONSTANT')
            last = tf.add(last, skip2, 'elementwise_sum')
            skip2 = last
            last = layers.apply_activation(last, activation=FLAGS.activation,
                                           data_format=FLAGS.data_format)
    # final dense layer
    l += 1
    print(last.get_shape())
    with tf.variable_scope('dense{}'.format(l)) as scope:
        last = tf.contrib.layers.flatten(last)
        last = tf.contrib.layers.fully_connected(last, FLAGS.num_labels, activation_fn=None)
    # return predicted labels
    print('Totally {} convolutional/fully-connected layers.'.format(l))
    return last

def train(total_loss, global_step):
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
    loss_averages_op = layers.loss_summaries(losses, FLAGS.loss_moving_average)
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
