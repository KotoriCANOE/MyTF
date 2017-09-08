import sys
import tensorflow as tf
sys.path.append('..')
from utils import layers

# basic parameters
tf.app.flags.DEFINE_string('data_format', 'NCHW', # 'NHWC'
                            """Data layout format.""")
tf.app.flags.DEFINE_integer('seq_size', 2048,
                            """Size of the 1-D sequence.""")
tf.app.flags.DEFINE_integer('num_labels', 12,
                            """Number of labels.""")

# model parameters - inference
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

# training parameters
tf.app.flags.DEFINE_integer('initializer', 3,
                            """Weights initialization method.""")
tf.app.flags.DEFINE_float('init_factor', 1.0,
                            """Weights initialization STD factor for conv layers without activation.""")
tf.app.flags.DEFINE_float('init_activation', 1.0,
                            """Weights initialization STD factor for conv layers with activation.""")
tf.app.flags.DEFINE_float('weight_decay', 1e-5,
                            """L2 regularization weight decay factor""")
tf.app.flags.DEFINE_float('learning_rate', 1e-4,
                            """Initial learning rate""")
tf.app.flags.DEFINE_float('lr_min', 1e-5,
                            """Minimum learning rate""")
tf.app.flags.DEFINE_float('lr_decay_steps', 500,
                            """Steps after which learning rate decays""")
tf.app.flags.DEFINE_float('lr_decay_factor', 0.99,
                            """Learning rate decay factor""")
tf.app.flags.DEFINE_float('learning_momentum', 0.9,
                            """momentum for MomentumOptimizer""")
tf.app.flags.DEFINE_float('learning_beta1', 0.9,
                            """beta1 for AdamOptimizer""")
tf.app.flags.DEFINE_float('learning_beta2', 0.999,
                            """beta2 for AdamOptimizer""")
tf.app.flags.DEFINE_float('epsilon', 1e-8,
                            """Fuzz term to avoid numerical instability""")
tf.app.flags.DEFINE_float('gradient_clipping', 0, #0.002,
                            """Gradient clipping factor""")
tf.app.flags.DEFINE_float('loss_moving_average', 0, #0.9,
                            """The decay to use for the moving average of losses""")
tf.app.flags.DEFINE_float('train_moving_average', 0, #0.9999,
                            """The decay to use for the moving average of trainable variables""")

# model
class MRSmodel(object):
    def __init__(self, config, data_format='NCHW',
                 seq_size=None, num_labels=None, batch_size=None):
        self.data_format = data_format
        self.seq_size = seq_size
        self.num_labels = num_labels
        self.batch_size = batch_size
        
        self.k_first = config.k_first
        self.res_blocks = config.res_blocks
        self.channels = config.channels
        self.batch_norm = config.batch_norm
        self.activation = config.activation
        
        self.initializer = config.initializer
        self.init_factor = config.init_factor
        self.init_activation = config.init_activation
        self.weight_decay = config.weight_decay
        self.learning_rate = config.learning_rate
        self.lr_min = config.lr_min
        self.lr_decay_steps = config.lr_decay_steps
        self.lr_decay_factor = config.lr_decay_factor
        self.learning_momentum = config.learning_momentum
        self.learning_beta1 = config.learning_beta1
        self.learning_beta2 = config.learning_beta2
        self.epsilon = config.epsilon
        self.gradient_clipping = config.gradient_clipping
        self.loss_moving_average = config.loss_moving_average
        self.train_moving_average = config.train_moving_average
        
        self.inference_weight_key = 'inference_weights'
        self.inference_loss_key = 'inference_losses'
        self.inference_total_loss_key = 'inference_total_loss'
        
        self.dtype = tf.float32
        self.shape_label = [self.batch_size, self.num_labels]
        if self.data_format == 'NCHW':
            self.shape = [self.batch_size, 1, 1, self.seq_size]
        else:
            self.shape = [self.batch_size, 1, self.seq_size, 1]
    
    def inference(self, spectrum, is_training=False, reuse=None):
        print('k_first={}, res_blocks={}, channels={}'.format(
            self.k_first, self.res_blocks, self.channels))
        # parameters
        data_format = self.data_format
        channels = self.channels
        batch_norm = self.batch_norm
        activation = self.activation
        initializer = self.initializer
        init_factor = self.init_factor
        init_activation = self.init_activation
        weight_key = self.inference_weight_key
        channel_index = -3 if self.data_format == 'NCHW' else -1
        pool_ksize = [1, 1, 1, 3] if self.data_format == 'NCHW' else [1, 1, 3, 1]
        reduce_strides = [1, 1, 1, 2] if self.data_format == 'NCHW' else [1, 1, 2, 1]
        # initialization
        last = spectrum
        l = 0
        with tf.variable_scope('inference', reuse=reuse) as scope:
            # first conv layer
            l += 1
            with tf.variable_scope('conv{}'.format(l)) as scope:
                last = layers.conv2d(last, ksize=[1, self.k_first], out_channels=channels,
                    stride=[1, 1, 1, 1], padding='SAME', data_format=data_format,
                    batch_norm=None, is_training=is_training, activation=activation,
                    initializer=initializer, init_factor=init_activation,
                    collection=weight_key)
            # residual blocks
            rb = 0
            skip2 = last
            while rb < self.res_blocks:
                rb += 1
                reduce_size = rb % 1 == 0
                strides = reduce_strides if reduce_size else [1, 1, 1, 1]
                double_channel = rb % 3 == 1
                if double_channel: channels *= 2
                l += 1
                with tf.variable_scope('conv{}'.format(l)) as scope:
                    last = layers.conv2d(last, ksize=[1, 1], out_channels=channels // 2,
                        stride=1, padding='SAME', data_format=data_format,
                        batch_norm=batch_norm, is_training=is_training, activation=activation,
                        initializer=initializer, init_factor=init_activation,
                        collection=weight_key)
                l += 1
                with tf.variable_scope('conv{}'.format(l)) as scope:
                    last = layers.conv2d(last, ksize=[1, 3], out_channels=channels // 2,
                        stride=strides, padding='SAME', data_format=data_format,
                        batch_norm=batch_norm, is_training=is_training, activation=activation,
                        initializer=initializer, init_factor=init_activation,
                        collection=weight_key)
                l += 1
                with tf.variable_scope('conv{}'.format(l)) as scope:
                    last = layers.conv2d(last, ksize=[1, 1], out_channels=channels,
                        stride=1, padding='SAME', data_format=data_format,
                        batch_norm=batch_norm, is_training=is_training, activation=None,
                        initializer=initializer, init_factor=init_factor,
                        collection=weight_key)
                with tf.variable_scope('skip_connection{}'.format(l)) as scope:
                    if reduce_size:
                        skip2 = tf.nn.avg_pool(skip2, ksize=reduce_strides, strides=reduce_strides,
                                               padding='SAME', data_format=data_format)
                    if double_channel:
                        padding = [[0, 0] for _ in range(4)]
                        padding[channel_index] = [0, channels // 2]
                        skip2 = tf.pad(skip2, padding, mode='CONSTANT')
                    last = tf.add(last, skip2)
                    skip2 = last
                    last = layers.apply_activation(last, activation=activation,
                                                   data_format=data_format)
            # final dense layer (num_labels)
            l += 1
            print(last.get_shape())
            with tf.variable_scope('dense{}'.format(l)) as scope:
                last = tf.contrib.layers.flatten(last)
                last = tf.contrib.layers.fully_connected(last, self.num_labels, activation_fn=None)
        # return predicted labels
        print('Totally {} convolutional/dense layers.'.format(l))
        return last
    
    def inference_losses(self, ref, pred):
        collection = self.inference_loss_key
        with tf.variable_scope('inference_losses') as scope:
            # L2 regularization weight decay
            if self.weight_decay > 0:
                with tf.variable_scope('l2_regularize') as scope:
                    l2_regularize = tf.add_n([tf.nn.l2_loss(v) for v in
                        tf.get_collection(self.inference_weight_key)])
                    l2_regularize = tf.multiply(l2_regularize, self.weight_decay, name='loss')
                    tf.losses.add_loss(l2_regularize, loss_collection=collection)
            '''
            # L2 loss
            mse = tf.losses.mean_squared_error(ref, pred, weights=1.0,
                loss_collection=collection, scope='MSE_loss')
            '''
            # L1 loss
            mad = tf.losses.absolute_difference(ref, pred, weights=1.0,
                loss_collection=collection, scope='MAD_loss')
            # return total loss
            return tf.add_n(tf.losses.get_losses(loss_collection=collection),
                            name=self.inference_total_loss_key)

    def build_model(self, spectrum=None, is_training=False):
        # set inputs
        if spectrum is None:
            self.spectrum = tf.placeholder(self.dtype, self.shape, name='Input')
        else:
            self.spectrum = tf.identity(spectrum, name='Input')
            self.spectrum.set_shape(self.shape)
        
        # apply generator to inputs
        self.labels_pred = self.inference(self.spectrum, is_training=is_training)
        
        # set outputs
        tf.identity(self.labels_pred, name='Output')
        
        # return inference results
        return self.labels_pred
    
    def build_train(self, spectrum=None, labels_ref=None):
        # reference outputs
        if labels_ref is None:
            self.labels_ref = tf.placeholder(self.dtype, self.shape_label, name='Label')
        else:
            self.labels_ref = tf.identity(labels_ref, name='Label')
            self.labels_ref.set_shape(self.shape_label)
        
        # build inference
        self.build_model(spectrum, is_training=True)
        
        # build inference losses
        self.g_loss = self.inference_losses(self.labels_ref, self.labels_pred)
        
        # trainable variables
        t_vars = tf.trainable_variables()
        self.g_vars = [var for var in t_vars if 'inference' in var.name]
        
        # return total loss(es)
        return self.g_loss
    
    def train(self, global_step):
        print('lr: {}, decay steps: {}, decay factor: {}, min: {}, weight decay: {}'.format(
            self.learning_rate, self.lr_decay_steps, self.lr_decay_factor,
            self.lr_min, self.weight_decay))
        
        # dependency need to be updated
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        
        # generate moving averages of all losses and associated summaries
        losses = [self.g_loss]
        losses += tf.losses.get_losses(loss_collection=self.inference_loss_key)
        loss_averages_op = layers.loss_summaries(losses, self.loss_moving_average)
        if loss_averages_op: update_ops.append(loss_averages_op)
        
        # decay the learning rate exponentially based on the number of steps
        g_lr = self.learning_rate
        if self.lr_decay_steps > 0 and self.lr_decay_factor != 1:
            g_lr = tf.train.exponential_decay(g_lr, global_step,
                self.lr_decay_steps, self.lr_decay_factor, staircase=True)
            g_lr = tf.maximum(self.lr_min, g_lr)
        tf.summary.scalar('g_learning_rate', g_lr)
        
        # compute gradients
        with tf.control_dependencies(update_ops):
            g_opt = tf.train.AdamOptimizer(g_lr, beta1=self.learning_beta1,
                beta2=self.learning_beta2, epsilon=self.epsilon)
            g_grads_and_vars = g_opt.compute_gradients(self.g_loss, var_list=self.g_vars)
        
        # gradient clipping
        if self.gradient_clipping > 0:
            g_clip_value = self.gradient_clipping / g_lr
            g_grads_and_vars = [(tf.clip_by_value(
                    grad, -g_clip_value, g_clip_value, name='g_gradient_clipping'
                    ), var) for grad, var in g_grads_and_vars]
        
        # training ops
        g_train_ops = []
        
        # apply gradient
        g_train_ops.append(g_opt.apply_gradients(g_grads_and_vars, global_step))
        
        # add histograms for trainable variables
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
        
        # add histograms for gradients
        for grad, var in g_grads_and_vars:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)
        
        # track the moving averages of all trainable variables
        if self.train_moving_average > 0:
            variable_averages = tf.train.ExponentialMovingAverage(
                    self.train_moving_average, global_step, name='train_moving_average')
            variable_averages_op = variable_averages.apply(tf.trainable_variables())
            g_train_ops.append(variable_averages_op)
        
        # generate operation
        with tf.control_dependencies(g_train_ops):
            g_train_op = tf.no_op(name='g_train')
        return g_train_op
