import sys
import tensorflow as tf
sys.path.append('..')
from utils import layers
import utils.image

# basic parameters
tf.app.flags.DEFINE_string('data_format', 'NCHW', # 'NHWC'
                            """Data layout format.""")
tf.app.flags.DEFINE_integer('input_range', 2,
                            """Internal used data range for input. Won't affect I/O. """
                            """1: [0, 1]; 2: [-1, 1]""")
tf.app.flags.DEFINE_integer('output_range', 2,
                            """Internal used data range for output. Won't affect I/O. """
                            """1: [0, 1]; 2: [-1, 1]""")
tf.app.flags.DEFINE_integer('qp_range', 0.05,
                            """Quantization parameter [0,1).""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")
tf.app.flags.DEFINE_boolean('multiGPU', False,
                            """Train the model using multiple GPUs.""")
tf.app.flags.DEFINE_integer('image_channels', 3,
                            """Channels of input/output image.""")

# model parameters - generator
tf.app.flags.DEFINE_integer('k_first', 3,
                            """Kernel size for the first layer.""")
tf.app.flags.DEFINE_integer('k_last', 3,
                            """Kernel size for the last layer.""")
tf.app.flags.DEFINE_integer('e_depth', 6,
                            """Depth of the network: number of layers, residual blocks, etc.""")
tf.app.flags.DEFINE_integer('d_depth', 6,
                            """Depth of the network: number of layers, residual blocks, etc.""")
tf.app.flags.DEFINE_integer('channels', 48,
                            """Number of features in hidden layers.""")
tf.app.flags.DEFINE_float('batch_norm', 0.999,
                            """Moving average decay for Batch Normalization.""")
tf.app.flags.DEFINE_string('activation', 'swish',
                            """Activation function used.""")

# training parameters
tf.app.flags.DEFINE_integer('initializer', 4,
                            """Weights initialization method.""")
tf.app.flags.DEFINE_float('init_factor', 1.0,
                            """Weights initialization STD factor for conv layers without activation.""")
tf.app.flags.DEFINE_float('init_activation', 2.0,
                            """Weights initialization STD factor for conv layers with activation.""")
tf.app.flags.DEFINE_float('weight_decay', 2e-6,
                            """L2 regularization weight decay factor""")
tf.app.flags.DEFINE_float('learning_rate', 1e-3,
                            """Initial learning rate""")
tf.app.flags.DEFINE_float('lr_min', 0,
                            """Minimum learning rate""")
tf.app.flags.DEFINE_float('lr_decay_steps', -200, #500,
                            """Steps after which learning rate decays""")
tf.app.flags.DEFINE_float('lr_decay_factor', 0.29, #0.01,
                            """Learning rate decay factor""")
tf.app.flags.DEFINE_float('learning_momentum', 0.9,
                            """momentum for MomentumOptimizer""")
tf.app.flags.DEFINE_float('learning_beta1', 0.9,
                            """beta1 for AdamOptimizer""")
tf.app.flags.DEFINE_float('learning_beta2', 0.999,
                            """beta2 for AdamOptimizer""")
tf.app.flags.DEFINE_float('epsilon', 1e-8,
                            """Fuzz term to avoid numerical instability""")
tf.app.flags.DEFINE_float('loss_moving_average', 0, #0.9,
                            """The decay to use for the moving average of losses""")
tf.app.flags.DEFINE_float('train_moving_average', 0.9999,
                            """The decay to use for the moving average of trainable variables""")

# model
class ICmodel(object):
    def __init__(self, config, data_format='NCHW', input_range=2, output_range=2,
                 qp_range=0.02, multiGPU=False, use_fp16=False, image_channels=3,
                 input_height=None, input_width=None, batch_size=None):
        self.data_format = data_format
        self.input_range = input_range
        self.output_range = output_range
        self.qp_range = qp_range
        self.multiGPU = multiGPU
        self.use_fp16 = use_fp16
        self.image_channels = image_channels
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = input_height
        self.output_width = input_width
        self.batch_size = batch_size
        
        self.k_first = config.k_first
        self.k_last = config.k_last
        self.e_depth = config.e_depth
        self.d_depth = config.d_depth
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
        self.loss_moving_average = config.loss_moving_average
        self.train_moving_average = config.train_moving_average
        
        self.generator_weight_key = 'generator_weights'
        self.generator_loss_key = 'generator_losses'
        self.generator_total_loss_key = 'generator_total_loss'
        
        self.dtype = tf.float16 if self.use_fp16 else tf.float32
        if self.data_format == 'NCHW':
            self.shape_src = [self.batch_size, self.image_channels, self.input_height, self.input_width]
            self.shape_dec = [self.batch_size, self.image_channels, self.output_height, self.output_width]
        else:
            self.shape_src = [self.batch_size, self.input_height, self.input_width, self.image_channels]
            self.shape_dec = [self.batch_size, self.output_height, self.output_width, self.image_channels]
    
    def generator(self, images_src, is_training=False, reuse=None):
        print('k_first={}, k_last={}, e_depth={}, d_depth={}, channels={}'.format(
            self.k_first, self.k_last, self.e_depth, self.d_depth, self.channels))
        # parameters
        data_format = self.data_format
        channels = self.channels
        batch_norm = self.batch_norm
        activation = self.activation
        initializer = self.initializer
        init_factor = self.init_factor
        init_activation = self.init_activation
        weight_key = self.generator_weight_key
        # initialization
        last = images_src
        l = 0
        with tf.variable_scope('generator', reuse=reuse) as scope:
            # encoder
            with tf.variable_scope('encoder') as scope:
                # encoder - first conv layer
                l += 1
                with tf.variable_scope('conv{}'.format(l)) as scope:
                    last = layers.conv2d(last, ksize=self.k_first, out_channels=channels,
                        stride=1, padding='SAME', data_format=data_format,
                        batch_norm=None, is_training=is_training, activation=None,
                        initializer=initializer, init_factor=init_activation,
                        collection=weight_key)
                # encoder - residual blocks
                depth = 0
                while depth < self.e_depth:
                    depth += 1
                    skip2 = last
                    downscale = False#depth == 4
                    if downscale:
                        channels *= 4
                        with tf.variable_scope('trans{}'.format(l)) as scope:
                            if data_format == 'NCHW':
                                last = utils.image.NCHW2NHWC(last)
                            last = tf.space_to_depth(last, 2)
                            if data_format == 'NCHW':
                                last = utils.image.NHWC2NCHW(last)
                        channels //= 2
                    l += 1
                    with tf.variable_scope('conv{}'.format(l)) as scope:
                        last = layers.apply_batch_norm(last, decay=batch_norm,
                            is_training=is_training, data_format=data_format)
                        last = layers.apply_activation(last, activation=activation,
                            data_format=data_format, collection=weight_key)
                        last = layers.conv2d(last, ksize=3, out_channels=channels,
                            stride=1, padding='SAME', data_format=data_format,
                            batch_norm=None, is_training=is_training, activation=None,
                            initializer=initializer, init_factor=init_activation,
                            collection=weight_key)
                    l += 1
                    with tf.variable_scope('conv{}'.format(l)) as scope:
                        last = layers.apply_batch_norm(last, decay=batch_norm,
                            is_training=is_training, data_format=data_format)
                        last = layers.apply_activation(last, activation=activation,
                            data_format=data_format, collection=weight_key)
                        last = layers.conv2d(last, ksize=3, out_channels=channels,
                            stride=1, padding='SAME', data_format=data_format,
                            batch_norm=None, is_training=is_training, activation=None,
                            initializer=initializer, init_factor=init_activation,
                            collection=weight_key)
                    with tf.variable_scope('skip_connection{}'.format(l)) as scope:
                        if downscale:
                            pool_size = [1, 1, 2, 2] if data_format == 'NCHW' else [1, 2, 2, 1]
                            skip2 = tf.nn.avg_pool(skip2, pool_size, pool_size,
                                padding='SAME', data_format=data_format)
                            padding = [[0, 0], [0, 0], [0, 0], [0, 0]]
                            padding[-3 if data_format == 'NCHW' else -1] = [0, channels // 2]
                            skip2 = tf.pad(skip2, padding)
                        last = layers.SqueezeExcitation(last, channel_r=1,
                            data_format=data_format, collection=weight_key)
                        last = tf.add(last, skip2)
                # encoder - final conv layer
                l += 1
                with tf.variable_scope('conv{}'.format(l)) as scope:
                    last = layers.conv2d(last, ksize=self.k_last, out_channels=self.image_channels,
                        stride=1, padding='SAME', data_format=data_format,
                        batch_norm=None, is_training=is_training, activation='tanh',
                        initializer=initializer, init_factor=init_factor,
                        collection=weight_key)
            # encoded images
            last = layers.quantize(last, [-1, 1], [0, 255], saturate=False)
            self.images_enc = last
            last = layers.convert_range(last, [0, 255], [-1, 1], saturate=False)
            # decoder
            with tf.variable_scope('decoder') as scope:
                # decoder - first conv layer
                channels //= 1
                l += 1
                with tf.variable_scope('conv{}'.format(l)) as scope:
                    last = layers.conv2d(last, ksize=self.k_first, out_channels=channels,
                        stride=1, padding='SAME', data_format=data_format,
                        batch_norm=None, is_training=is_training, activation=None,
                        initializer=initializer, init_factor=init_activation,
                        collection=weight_key)
                # decoder - residual blocks
                depth = 0
                skip2 = last
                while depth < self.d_depth:
                    depth += 1
                    l += 1
                    with tf.variable_scope('conv{}'.format(l)) as scope:
                        last = layers.apply_batch_norm(last, decay=batch_norm,
                            is_training=is_training, data_format=data_format)
                        last = layers.apply_activation(last, activation=activation,
                            data_format=data_format, collection=weight_key)
                        last = layers.conv2d(last, ksize=3, out_channels=channels,
                            stride=1, padding='SAME', data_format=data_format,
                            batch_norm=None, is_training=is_training, activation=None,
                            initializer=initializer, init_factor=init_activation,
                            collection=weight_key)
                    l += 1
                    with tf.variable_scope('conv{}'.format(l)) as scope:
                        last = layers.apply_batch_norm(last, decay=batch_norm,
                            is_training=is_training, data_format=data_format)
                        last = layers.apply_activation(last, activation=activation,
                            data_format=data_format, collection=weight_key)
                        last = layers.conv2d(last, ksize=3, out_channels=channels,
                            stride=1, padding='SAME', data_format=data_format,
                            batch_norm=None, is_training=is_training, activation=None,
                            initializer=initializer, init_factor=init_activation,
                            collection=weight_key)
                    with tf.variable_scope('skip_connection{}'.format(l)) as scope:
                        last = layers.SqueezeExcitation(last, channel_r=1,
                            data_format=data_format, collection=weight_key)
                        last = tf.add(last, skip2)
                '''
                # sub-pixel conv layer
                channels //= 2
                l += 1
                with tf.variable_scope('subpixel_conv{}'.format(l)) as scope:
                    last = layers.subpixel_conv2d(last, ksize=3, out_channels=channels,
                        scaling=2, padding='SAME', data_format=data_format,
                        batch_norm=None, is_training=is_training, activation=activation,
                        initializer=initializer, init_factor=init_activation,
                        collection=weight_key)
                '''
                # decoder - final conv layer
                l += 1
                with tf.variable_scope('conv{}'.format(l)) as scope:
                    last = layers.conv2d(last, ksize=self.k_last, out_channels=self.image_channels,
                        stride=1, padding='SAME', data_format=data_format,
                        batch_norm=None, is_training=is_training, activation=None,
                        initializer=initializer, init_factor=init_factor,
                        collection=weight_key)
        # return SR image
        print('Generator: totally {} convolutional layers.'.format(l))
        return last
    
    def generator_losses(self, ref, pred, enc, alpha=0.10, weights1=1.0, weights2=1.0):
        import utils.image
        collection = self.generator_loss_key
        with tf.variable_scope('generator_losses') as scope:
            # data range conversion
            if self.output_range == 2:
                ref = (ref + 1) * 0.5
                pred = (pred + 1) * 0.5
            # L2 regularization weight decay
            if self.weight_decay > 0:
                with tf.variable_scope('l2_regularize') as scope:
                    l2_regularize = tf.add_n([tf.nn.l2_loss(v) for v in
                        tf.get_collection(self.generator_weight_key)])
                    l2_regularize = tf.multiply(l2_regularize, self.weight_decay, name='loss')
                    tf.losses.add_loss(l2_regularize, loss_collection=collection)
            # quantization entropy
            entropy = layers.entropy(enc, [0, 256], 256, saturate=False)
            tf.losses.add_loss(entropy, loss_collection=collection)
            # L1 loss
            weights1 *= 1 - alpha
            weights2 *= alpha
            if alpha != 1.0:
                RGB_mad = tf.losses.absolute_difference(ref, pred,
                    weights=weights1, loss_collection=collection, scope='RGB_MAD_loss')
            # MS-SSIM: OPP color space - Y
            if alpha != 0.0:
                Y_ref = utils.image.RGB2Y(ref, data_format=self.data_format)
                Y_pred = utils.image.RGB2Y(pred, data_format=self.data_format)
                Y_ms_ssim = (1 - utils.image.MS_SSIM2(Y_ref, Y_pred, sigma=[0.6,1.5,4],
                            norm=False, data_format=self.data_format))
                Y_ms_ssim = tf.multiply(Y_ms_ssim, weights2, name='Y_MS_SSIM_loss')
                tf.losses.add_loss(Y_ms_ssim, loss_collection=collection)
            # return total loss
            return tf.add_n(tf.losses.get_losses(loss_collection=collection),
                            name=self.generator_total_loss_key)
    
    def build_model(self, images_src=None, is_training=False):
        # set inputs
        if images_src is None:
            self.images_src = tf.placeholder(self.dtype, self.shape_src, name='Source')
        else:
            self.images_src = tf.identity(images_src, name='Source')
            self.images_src.set_shape(self.shape_src)
        if self.input_range == 2:
            self.images_src = self.images_src * 2 - 1
        
        # apply generator to inputs
        self.images_dec = self.generator(self.images_src, is_training=is_training)
        
        # encoded images
        tf.cast(self.images_enc, tf.uint8, name='Encoded')

        # restore [0, 1] range for generated outputs
        if self.output_range == 2:
            tf.multiply(self.images_dec + 1, 0.5, name='Decoded')
        else:
            tf.identity(self.images_dec, name='Decoded')
        
        # trainable and model variables
        self.g_tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        self.g_mvars = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope='generator')
        self.g_svars = list(set(self.g_tvars + self.g_mvars))
        
        # track the moving averages of all trainable variables
        if not is_training and self.train_moving_average > 0:
            with tf.variable_scope('train_moving_average') as scope:
                ema = tf.train.ExponentialMovingAverage(self.train_moving_average)
                g_ema_op = ema.apply(self.g_svars)
                self.g_svars = {ema.average_name(var): var for var in self.g_svars}
        
        # return generated results
        return self.images_enc, self.images_dec
    
    def build_train(self, images_src=None):
        # build generator
        self.build_model(images_src, is_training=True)
        
        # build generator losses
        self.g_loss = self.generator_losses(self.images_src, self.images_dec, self.images_enc)
        
        # set learning rate
        self.g_lr = tf.Variable(self.learning_rate, trainable=False, name='generator_lr')
        
        # return total loss(es)
        return self.g_loss
    
    def lr_decay(self):
        self.g_lr_last = tf.Variable(self.g_lr, trainable=False)
        g_lr_last_op = tf.assign(self.g_lr_last, self.g_lr, use_locking=True)
        with tf.control_dependencies([g_lr_last_op]):
            g_lr_decay_op = tf.assign(self.g_lr, self.g_lr * (1 - self.lr_decay_factor), use_locking=True)
        return g_lr_decay_op
    
    def train(self, global_step):
        print('lr: {}, decay steps: {}, decay factor: {}, min: {}, weight decay: {}'.format(
            self.learning_rate, self.lr_decay_steps, self.lr_decay_factor,
            self.lr_min, self.weight_decay))
        
        # training ops
        g_train_ops = []
        
        # dependency need to be updated
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        
        # generate moving averages of all losses and associated summaries
        losses = [self.g_loss]
        losses += tf.losses.get_losses(loss_collection=self.generator_loss_key)
        loss_averages_op = layers.loss_summaries(losses, self.loss_moving_average)
        if loss_averages_op: update_ops.append(loss_averages_op)
        
        # decay the learning rate exponentially based on the number of steps
        if self.lr_decay_steps > 0 and self.lr_decay_factor != 0:
            g_lr = tf.train.exponential_decay(self.g_lr, global_step,
                self.lr_decay_steps, 1 - self.lr_decay_factor, staircase=True)
            if self.lr_min > 0:
                g_lr = tf.maximum(tf.constant(self.lr_min, dtype=self.dtype), g_lr)
        else:
            g_lr = self.g_lr
        tf.summary.scalar('g_learning_rate', g_lr)
        
        # optimizer
        g_opt = tf.contrib.opt.NadamOptimizer(g_lr, beta1=self.learning_beta1,
            beta2=self.learning_beta2, epsilon=self.epsilon)
        
        # compute gradients
        with tf.control_dependencies(update_ops):
            g_grads_and_vars = g_opt.compute_gradients(self.g_loss, self.g_tvars)
        
        # apply gradients
        g_train_ops.append(g_opt.apply_gradients(g_grads_and_vars, global_step))
        
        # add histograms for gradients
        for grad, var in g_grads_and_vars:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)
            if var is not None:
                tf.summary.histogram(var.op.name, var)
        
        # track the moving averages of all trainable variables
        if self.train_moving_average > 0:
            with tf.variable_scope('train_moving_average') as scope:
                ema = tf.train.ExponentialMovingAverage(self.train_moving_average, global_step)
                g_ema_op = ema.apply(self.g_svars)
                g_train_ops.append(g_ema_op)
                self.g_rvars = {ema.average_name(var): var for var in self.g_svars}
                self.g_svars = [ema.average(var) for var in self.g_svars]
        
        # generate operation
        with tf.control_dependencies(g_train_ops):
            g_train_op = tf.no_op(name='g_train')
        return g_train_op
