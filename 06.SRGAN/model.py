import sys
import tensorflow as tf
sys.path.append('..')
from utils import helper
from utils import layers

# basic parameters
tf.app.flags.DEFINE_string('data_format', 'NCHW', # 'NHWC'
                            """Data layout format.""")
tf.app.flags.DEFINE_integer('input_range', 2,
                            """Internal used data range for input. Won't affect I/O. """
                            """1: [0, 1]; 2: [-1, 1]""")
tf.app.flags.DEFINE_integer('output_range', 2,
                            """Internal used data range for output. Won't affect I/O. """
                            """1: [0, 1]; 2: [-1, 1]""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")
tf.app.flags.DEFINE_boolean('multiGPU', False,
                            """Train the model using multiple GPUs.""")
tf.app.flags.DEFINE_integer('scaling', 2,
                            """Scaling ratio of the super-resolution filter.""")
tf.app.flags.DEFINE_integer('image_channels', 3,
                            """Channels of input/output image.""")

# model parameters - generator
tf.app.flags.DEFINE_integer('k_first', 3,
                            """Kernel size for the first layer.""")
tf.app.flags.DEFINE_integer('k_last', 3,
                            """Kernel size for the last layer.""")
tf.app.flags.DEFINE_integer('res_blocks', 8,
                            """Number of residual blocks.""")
tf.app.flags.DEFINE_integer('channels', 64,
                            """Number of features in hidden layers.""")
tf.app.flags.DEFINE_integer('channels2', 32,
                            """Number of features after resize conv.""")
tf.app.flags.DEFINE_float('batch_norm', 0, #0.999,
                            """Moving average decay for Batch Normalization.""")
tf.app.flags.DEFINE_string('activation', 'prelu',
                            """Activation function used.""")

# model parameters - discriminator
tf.app.flags.DEFINE_integer('d_blocks', 3,
                            """Number of blocks.""")
tf.app.flags.DEFINE_integer('d_channels', 64,
                            """Number of features in hidden layers.""")
tf.app.flags.DEFINE_float('d_batch_norm', 0, #0.999,
                            """Moving average decay for Batch Normalization.""")
tf.app.flags.DEFINE_string('d_activation', 'lrelu0.2',
                            """Activation function used.""")

# training parameters
tf.app.flags.DEFINE_integer('initializer', 5,
                            """Weights initialization method.""")
tf.app.flags.DEFINE_float('init_factor', 1.0,
                            """Weights initialization STD factor for conv layers without activation.""")
tf.app.flags.DEFINE_float('init_activation', 1.0,
                            """Weights initialization STD factor for conv layers with activation.""")
tf.app.flags.DEFINE_float('weight_decay', 1e-5,
                            """L2 regularization weight decay factor""")
tf.app.flags.DEFINE_float('learning_rate', 1e-4,
                            """Initial learning rate""")
tf.app.flags.DEFINE_float('lr_min', 1e-8,
                            """Minimum learning rate""")
tf.app.flags.DEFINE_float('lr_decay_steps', 1e3,
                            """Steps after which learning rate decays""")
tf.app.flags.DEFINE_float('lr_decay_factor', 0.98,
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

# training parameters - discriminator
tf.app.flags.DEFINE_float('d_weight_decay', 1e-5,
                            """L2 regularization weight decay factor""")
tf.app.flags.DEFINE_float('d_learning_rate', 1e-4,
                            """Initial learning rate""")
tf.app.flags.DEFINE_float('d_lr_min', 1e-8,
                            """Minimum learning rate""")

# model
class SRmodel(object):
    def __init__(self, config, data_format='NCHW', input_range=1, output_range=1,
                 multiGPU=False, use_fp16=False, scaling=2, image_channels=3,
                 input_height=None, input_width=None, batch_size=None):
        self.data_format = data_format
        self.input_range = input_range
        self.output_range = output_range
        self.multiGPU = multiGPU
        self.use_fp16 = use_fp16
        self.scaling = scaling
        self.image_channels = image_channels
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = input_height * scaling if input_height else None
        self.output_width = input_width * scaling if input_width else None
        self.batch_size = batch_size
        
        self.k_first = config.k_first
        self.k_last = config.k_last
        self.res_blocks = config.res_blocks
        self.channels = config.channels
        self.channels2 = config.channels2
        self.batch_norm = config.batch_norm
        self.activation = config.activation
        
        self.d_blocks = config.d_blocks
        self.d_channels = config.d_channels
        self.d_batch_norm = config.d_batch_norm
        self.d_activation = config.d_activation
        
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
        
        self.d_weight_decay = config.d_weight_decay
        self.d_learning_rate = config.d_learning_rate
        self.d_lr_min = config.d_lr_min
        
        self.generator_weight_key = 'generator_weights'
        self.generator_loss_key = 'generator_losses'
        self.generator_total_loss_key = 'generator_total_loss'
        self.discriminator_weight_key = 'discriminator_weights'
        self.discriminator_loss_key = 'discriminator_losses'
        self.discriminator_total_loss_key = 'discriminator_total_loss'
        
        self.dtype = tf.float16 if self.use_fp16 else tf.float32
        if self.data_format == 'NCHW':
            self.shape_lr = [self.batch_size, self.image_channels, self.input_height, self.input_width]
            self.shape_hr = [self.batch_size, self.image_channels, self.output_height, self.output_width]
        else:
            self.shape_lr = [self.batch_size, self.input_height, self.input_width, self.image_channels]
            self.shape_hr = [self.batch_size, self.output_height, self.output_width, self.image_channels]
    
    def generator(self, images_lr, is_training=False, reuse=None):
        print('k_first={}, k_last={}, res_blocks={}, channels={}, channels2={}'.format(
            self.k_first, self.k_last, self.res_blocks, self.channels, self.channels2))
        # parameters
        data_format = self.data_format
        channels = self.channels
        channels2 = self.channels2
        batch_norm = self.batch_norm
        activation = self.activation
        initializer = self.initializer
        init_factor = self.init_factor
        init_activation = self.init_activation
        weight_key = self.generator_weight_key
        # initialization
        last = images_lr
        l = 0
        with tf.variable_scope('generator', reuse=reuse) as scope:
            # first conv layer
            l += 1
            with tf.variable_scope('conv{}'.format(l)) as scope:
                last = layers.conv2d(last, ksize=self.k_first, out_channels=channels,
                    stride=1, padding='SAME', data_format=data_format,
                    batch_norm=None, is_training=is_training, activation=activation,
                    initializer=initializer, init_factor=init_activation,
                    collection=weight_key)
            skip1 = last
            # residual blocks
            rb = 0
            skip2 = last
            while rb < self.res_blocks:
                rb += 1
                l += 1
                with tf.variable_scope('conv{}'.format(l)) as scope:
                    last = layers.conv2d(last, ksize=3, out_channels=channels,
                        stride=1, padding='SAME', data_format=data_format,
                        batch_norm=batch_norm, is_training=is_training, activation=activation,
                        initializer=initializer, init_factor=init_activation,
                        collection=weight_key)
                l += 1
                with tf.variable_scope('conv{}'.format(l)) as scope:
                    last = layers.conv2d(last, ksize=3, out_channels=channels,
                        stride=1, padding='SAME', data_format=data_format,
                        batch_norm=batch_norm, is_training=is_training, activation=None,
                        initializer=initializer, init_factor=init_factor,
                        collection=weight_key)
                with tf.variable_scope('skip_connection{}'.format(l)) as scope:
                    last = tf.add(last, skip2, 'elementwise_sum')
                    skip2 = last
                    last = layers.apply_activation(last, activation=activation,
                                                   data_format=data_format)
            # skip connection
            l += 1
            with tf.variable_scope('conv{}'.format(l)) as scope:
                last = layers.conv2d(last, ksize=3, out_channels=channels,
                    stride=1, padding='SAME', data_format=data_format,
                    batch_norm=batch_norm, is_training=is_training, activation=None,
                    initializer=initializer, init_factor=init_factor,
                    collection=weight_key)
            with tf.variable_scope('skip_connection{}'.format(l)) as scope:
                last = tf.add(last, skip1, 'elementwise_sum')
                last = layers.apply_activation(last, activation=activation,
                                               data_format=data_format)
            # resize conv layer
            l += 1
            with tf.variable_scope('resize_conv{}'.format(l)) as scope:
                last = layers.resize_conv2d(last, ksize=3, out_channels=channels2,
                    scaling=self.scaling, data_format=data_format,
                    batch_norm=None, is_training=is_training, activation=activation,
                    initializer=initializer, init_factor=init_activation,
                    collection=weight_key)
            '''
            # sub-pixel conv layer
            l += 1
            with tf.variable_scope('subpixel_conv{}'.format(l)) as scope:
                last = layers.subpixel_conv2d(last, ksize=3, out_channels=channels2,
                    scaling=self.scaling, data_format=data_format,
                    batch_norm=None, is_training=is_training, activation=activation,
                    initializer=initializer, init_factor=init_activation,
                    collection=weight_key)
            '''
            # final conv layer
            l += 1
            with tf.variable_scope('conv{}'.format(l)) as scope:
                last = layers.conv2d(last, ksize=self.k_last, out_channels=self.image_channels,
                    stride=1, padding='SAME', data_format=data_format,
                    batch_norm=None, is_training=is_training, activation=None,
                    initializer=initializer, init_factor=init_factor,
                    collection=weight_key)
        # return SR image
        print('Totally {} convolutional layers.'.format(l))
        return last
    
    def discriminator(self, images, is_training=False, reuse=None):
        # parameters
        data_format = self.data_format
        channels = self.d_channels
        batch_norm = self.d_batch_norm
        activation = self.d_activation
        initializer = self.initializer
        init_factor = self.init_factor
        init_activation = self.init_activation
        weight_key = self.discriminator_weight_key
        # initialization
        last = images
        l = 0
        with tf.variable_scope('discriminator', reuse=reuse) as scope:
            # first conv layer
            l += 1
            with tf.variable_scope('conv{}'.format(l)) as scope:
                last = layers.conv2d(last, ksize=3, out_channels=channels,
                    stride=1, padding='SAME', data_format=data_format,
                    batch_norm=None, is_training=is_training, activation=activation,
                    initializer=initializer, init_factor=init_activation,
                    collection=weight_key)
            # blocks
            b = 0
            skip2 = last
            while b < self.d_blocks:
                b += 1
                l += 1
                with tf.variable_scope('conv{}'.format(l)) as scope:
                    last = layers.conv2d(last, ksize=3, out_channels=channels,
                        stride=2, padding='SAME', data_format=data_format,
                        batch_norm=batch_norm, is_training=is_training, activation=activation,
                        initializer=initializer, init_factor=init_activation,
                        collection=weight_key)
                l += 1
                with tf.variable_scope('conv{}'.format(l)) as scope:
                    last = layers.conv2d(last, ksize=3, out_channels=channels,
                        stride=1, padding='SAME', data_format=data_format,
                        batch_norm=batch_norm, is_training=is_training, activation=activation,
                        initializer=initializer, init_factor=init_activation,
                        collection=weight_key)
                channels *= 2
            # final conv layer
            l += 1
            with tf.variable_scope('conv{}'.format(l)) as scope:
                last = layers.conv2d(last, ksize=3, out_channels=channels,
                    stride=2, padding='SAME', data_format=data_format,
                    batch_norm=batch_norm, is_training=is_training, activation=activation,
                    initializer=initializer, init_factor=init_activation,
                    collection=weight_key)
            # dense layer (1024)
            l += 1
            print(last.get_shape())
            with tf.variable_scope('dense{}'.format(l)) as scope:
                last = tf.contrib.layers.flatten(last)
                last = tf.contrib.layers.fully_connected(last, 1024, activation_fn=None)
                last = layers.apply_activation(last, activation=activation,
                                               data_format=data_format)
            # dense layer (1)
            l += 1
            with tf.variable_scope('dense{}'.format(l)) as scope:
                last = tf.contrib.layers.fully_connected(last, 1, activation_fn=None)
        # return discriminating logits
        return last
    
    def generator_losses(self, gtruth, pred, pred_logits, alpha=0.0, weights1=1.0, weights2=1.0):
        import utils.image
        collection = self.generator_loss_key
        with tf.variable_scope('generator_losses') as scope:
            # adversarial loss
            g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=pred_logits, labels=tf.ones_like(pred_logits)))
            tf.summary.scalar('g_loss', g_loss)
            ad_loss = tf.multiply(g_loss, 1e-2, name='adversarial_loss')
            tf.losses.add_loss(ad_loss, loss_collection=collection)
            # data range conversion
            if self.output_range == 2:
                gtruth = (gtruth + 1) * 0.5
                pred = (pred + 1) * 0.5
            # L2 regularization weight decay
            if self.weight_decay > 0:
                with tf.variable_scope('l2_regularize') as scope:
                    l2_regularize = tf.add_n([tf.nn.l2_loss(v) for v in
                        tf.get_collection(self.generator_weight_key)])
                    l2_regularize = tf.multiply(l2_regularize, self.weight_decay, name='loss')
                    tf.losses.add_loss(l2_regularize, loss_collection=collection)
            # L1 loss
            weights1 *= 1 - alpha
            weights2 *= alpha
            if alpha != 1.0:
                RGB_mad = tf.losses.absolute_difference(gtruth, pred,
                    weights=weights1, loss_collection=collection, scope='RGB_MAD_loss')
            # MS-SSIM: OPP color space - Y
            if alpha != 0.0:
                Y_gtruth = utils.image.RGB2Y(gtruth, data_format=self.data_format)
                Y_pred = utils.image.RGB2Y(pred, data_format=self.data_format)
                Y_ms_ssim = (1 - utils.image.MS_SSIM2(Y_gtruth, Y_pred, sigma=[0.6,1.5,4],
                            norm=False, data_format=self.data_format))
                Y_ms_ssim = tf.multiply(Y_ms_ssim, weights2, name='Y_MS_SSIM_loss')
                tf.losses.add_loss(Y_ms_ssim, loss_collection=collection)
            # return total loss
            return tf.add_n(tf.losses.get_losses(loss_collection=collection),
                            name=self.generator_total_loss_key)
    
    def discriminator_losses(self, gtruth_logits, pred_logits):
        collection = self.discriminator_loss_key
        with tf.variable_scope('discriminator_losses') as scope:
            # L2 regularization weight decay
            if self.d_weight_decay > 0:
                with tf.variable_scope('l2_regularize') as scope:
                    l2_regularize = tf.add_n([tf.nn.l2_loss(v) for v in
                        tf.get_collection(self.discriminator_weight_key)])
                    l2_regularize = tf.multiply(l2_regularize, self.d_weight_decay, name='loss')
                    tf.losses.add_loss(l2_regularize, loss_collection=collection)
            # adversarial loss
            d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=gtruth_logits, labels=tf.ones_like(gtruth_logits)))
            d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=pred_logits, labels=tf.zeros_like(pred_logits)))
            tf.summary.scalar('d_loss_real', d_loss_real)
            tf.summary.scalar('d_loss_fake', d_loss_fake)
            ad_loss = tf.add(d_loss_real, d_loss_fake, name='adversarial_loss')
            tf.losses.add_loss(ad_loss, loss_collection=collection)
            # return total loss
            return tf.add_n(tf.losses.get_losses(loss_collection=collection),
                            name=self.discriminator_total_loss_key)
    
    def build_model(self, images_lr=None, is_training=False):
        # input samples
        if images_lr is None:
            self.images_lr = tf.placeholder(self.dtype, self.shape_lr, name='Input')
        else:
            self.images_lr = tf.identity(images_lr, name='Input')
            self.images_lr.set_shape(self.shape_lr)
        if self.input_range == 2:
            self.images_lr = self.images_lr * 2 - 1
        
        # apply generator to input samples
        self.images_sr = self.generator(self.images_lr, is_training=is_training)
        
        # generated output samples
        self.images_sr.set_shape(self.shape_hr)
        
        # restore [0, 1] range for generated output samples
        if self.output_range == 2:
            tf.multiply(self.images_sr + 1, 0.5, name='Output')
        else:
            tf.identity(self.images_sr, name='Output')
        
        # return generated samples
        return self.images_sr
    
    def build_train(self, images_lr=None, images_hr=None):
        # output samples - from data generating distribution
        if images_hr is None:
            self.images_hr = tf.placeholder(self.dtype, self.shape_hr, name='Label')
        else:
            self.images_hr = tf.identity(images_hr, name='Label')
            self.images_hr.set_shape(self.shape_hr)
        if self.output_range == 2:
            self.images_hr = self.images_hr * 2 - 1
        
        # build generator
        self.build_model(images_lr, is_training=True)
        
        # build discriminator
        d_logits_hr = self.discriminator(self.images_hr, is_training=True)
        d_logits_sr = self.discriminator(self.images_sr, is_training=True, reuse=True)
        
        # build generator losses
        self.g_loss = self.generator_losses(self.images_hr, self.images_sr, d_logits_sr)
        
        # build discriminator losses
        self.d_loss = self.discriminator_losses(d_logits_hr, d_logits_sr)
        
        # trainable variables
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]
        
        # return total loss(es)
        return self.g_loss, self.d_loss
    
    def train(self, global_step):
        print('lr: {}|{}, decay steps: {}, decay factor: {}, min: {}|{}, weight decay: {}|{}'.format(
            self.learning_rate, self.d_learning_rate, self.lr_decay_steps, self.lr_decay_factor,
            self.lr_min, self.d_lr_min, self.weight_decay, self.d_weight_decay))
        
        # dependency need to be updated
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        
        # generate moving averages of all losses and associated summaries
        losses = [self.g_loss, self.d_loss]
        losses += tf.losses.get_losses(loss_collection=self.generator_loss_key)
        losses += tf.losses.get_losses(loss_collection=self.discriminator_loss_key)
        loss_averages_op = layers.loss_summaries(losses, self.loss_moving_average)
        if loss_averages_op: update_ops.append(loss_averages_op)
        
        # decay the learning rate exponentially based on the number of steps
        d_lr = self.d_learning_rate
        g_lr = self.learning_rate
        if self.lr_decay_steps > 0 and self.lr_decay_factor != 1:
            d_lr = tf.train.exponential_decay(d_lr, global_step,
                self.lr_decay_steps, self.lr_decay_factor, staircase=True)
            d_lr = tf.maximum(self.d_lr_min, d_lr)
            g_lr = tf.train.exponential_decay(g_lr, global_step,
                self.lr_decay_steps, self.lr_decay_factor, staircase=True)
            g_lr = tf.maximum(self.lr_min, g_lr)
        tf.summary.scalar('d_learning_rate', d_lr)
        tf.summary.scalar('g_learning_rate', g_lr)
        
        # compute gradients
        with tf.control_dependencies(update_ops):
            d_opt = tf.train.AdamOptimizer(d_lr, beta1=self.learning_beta1,
                beta2=self.learning_beta2, epsilon=self.epsilon)
            g_opt = tf.train.AdamOptimizer(g_lr, beta1=self.learning_beta1,
                beta2=self.learning_beta2, epsilon=self.epsilon)
            d_grads_and_vars = d_opt.compute_gradients(self.d_loss, var_list=self.d_vars)
            g_grads_and_vars = g_opt.compute_gradients(self.g_loss, var_list=self.g_vars)
        
        # gradient clipping
        if self.gradient_clipping > 0:
            d_clip_value = self.gradient_clipping / d_lr
            g_clip_value = self.gradient_clipping / g_lr
            d_grads_and_vars = [(tf.clip_by_value(
                    grad, -d_clip_value, d_clip_value, name='d_gradient_clipping'
                    ), var) for grad, var in d_grads_and_vars]
            g_grads_and_vars = [(tf.clip_by_value(
                    grad, -g_clip_value, g_clip_value, name='g_gradient_clipping'
                    ), var) for grad, var in g_grads_and_vars]
        
        # training ops
        train_ops = []
        
        # apply gradient
        train_ops.append(d_opt.apply_gradients(d_grads_and_vars, None))
        train_ops.append(g_opt.apply_gradients(g_grads_and_vars, global_step))
        
        # add histograms for trainable variables
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
        
        # add histograms for gradients
        for grad, var in g_grads_and_vars + d_grads_and_vars:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)
        
        # track the moving averages of all trainable variables
        if self.train_moving_average > 0:
            variable_averages = tf.train.ExponentialMovingAverage(
                    self.train_moving_average, global_step, name='train_moving_average')
            variable_averages_op = variable_averages.apply(tf.trainable_variables())
            train_ops.append(variable_averages_op)
        
        # generate operation
        with tf.control_dependencies(train_ops):
            train_op = tf.no_op(name='train')
        return train_op
