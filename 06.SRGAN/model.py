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
tf.app.flags.DEFINE_integer('g_depth', 8,
                            """Depth of the network: number of layers, residual blocks, etc.""")
tf.app.flags.DEFINE_integer('channels', 80,
                            """Number of features in hidden layers.""")
tf.app.flags.DEFINE_integer('channels2', 40,
                            """Number of features after resize conv.""")
tf.app.flags.DEFINE_float('batch_norm', 0.99,
                            """Moving average decay for Batch Normalization.""")
tf.app.flags.DEFINE_string('activation', 'su',
                            """Activation function used.""")

# model parameters - discriminator
tf.app.flags.DEFINE_integer('d_depth', 3,
                            """Depth of the network: number of layers, residual blocks, etc.""")
tf.app.flags.DEFINE_integer('d_channels', 48,
                            """Number of features in hidden layers.""")
tf.app.flags.DEFINE_float('d_batch_norm', 0, #0.999,
                            """Moving average decay for Batch Normalization.""")
tf.app.flags.DEFINE_string('d_activation', 'lrelu0.2',
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
tf.app.flags.DEFINE_float('learning_rate', 1e-4,
                            """Initial learning rate""")
tf.app.flags.DEFINE_float('lr_min', 0,
                            """Minimum learning rate""")
tf.app.flags.DEFINE_float('lr_decay_steps', 200,
                            """Steps after which learning rate decays""")
tf.app.flags.DEFINE_float('lr_decay_factor', 0.01,
                            """Learning rate decay factor""")
tf.app.flags.DEFINE_float('learning_momentum', 0.9,
                            """momentum for MomentumOptimizer""")
tf.app.flags.DEFINE_float('learning_beta1', 0.5,
                            """beta1 for AdamOptimizer""")
tf.app.flags.DEFINE_float('learning_beta2', 0.9,
                            """beta2 for AdamOptimizer""")
tf.app.flags.DEFINE_float('epsilon', 1e-8,
                            """Fuzz term to avoid numerical instability""")
tf.app.flags.DEFINE_float('loss_moving_average', 0, #0.9,
                            """The decay to use for the moving average of losses""")
tf.app.flags.DEFINE_float('train_moving_average', 0.9999,
                            """The decay to use for the moving average of trainable variables""")

# training parameters - discriminator
tf.app.flags.DEFINE_float('d_weight_decay', 1e-6,
                            """L2 regularization weight decay factor""")
tf.app.flags.DEFINE_float('d_learning_rate', 1e-4,
                            """Initial learning rate""")
tf.app.flags.DEFINE_float('d_lr_min', 0,
                            """Minimum learning rate""")
tf.app.flags.DEFINE_integer('gan_loss', 2,
                            """Loss function for GAN training\n"""
                            """1: DCGAN;\n"""
                            """2: WGAN-GP;\n"""
                            """3: WGAN-GP using Hessian-Vector products trick;\n"""
                            """4: WGAN-GP using finite difference of 2 random samples;\n"""
                            """5: WGAN-GP using difference of real and fake samples;""")
tf.app.flags.DEFINE_float('lp_lambda', 10,
                          """Gradient penalty lambda hyperparameter""")
tf.app.flags.DEFINE_integer('critic_iters', 5,
                            """Number of discriminator training iters every generator training iter""")

# model
class SRmodel(object):
    def __init__(self, config, data_format='NCHW', input_range=2, output_range=2,
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
        self.g_depth = config.g_depth
        self.channels = config.channels
        self.channels2 = config.channels2
        self.batch_norm = config.batch_norm
        self.activation = config.activation
        
        self.d_depth = config.d_depth
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
        self.loss_moving_average = config.loss_moving_average
        self.train_moving_average = config.train_moving_average
        
        self.d_weight_decay = config.d_weight_decay
        self.d_learning_rate = config.d_learning_rate
        self.d_lr_min = config.d_lr_min
        self.gan_loss = config.gan_loss
        self.lp_lambda = config.lp_lambda
        
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
        print('k_first={}, k_last={}, g_depth={}, channels={}, channels2={}'.format(
            self.k_first, self.k_last, self.g_depth, self.channels, self.channels2))
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
            skip0 = last
            # first conv layer
            l += 1
            with tf.variable_scope('conv{}'.format(l)) as scope:
                last = layers.conv2d(last, ksize=self.k_first, out_channels=channels,
                    stride=1, padding='SAME', data_format=data_format,
                    batch_norm=None, is_training=is_training, activation=None,
                    initializer=initializer, init_factor=init_activation,
                    collection=weight_key)
            skip1 = last
            # residual blocks
            depth = 0
            skip2 = last
            while depth < self.g_depth:
                depth += 1
                l += 1
                with tf.variable_scope('conv{}'.format(l)) as scope:
                    last = layers.apply_batch_norm(last, decay=batch_norm,
                        is_training=is_training, data_format=data_format)
                    last = layers.apply_activation(last, activation=activation,
                        data_format=data_format)
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
                        data_format=data_format)
                    last = layers.conv2d(last, ksize=3, out_channels=channels,
                        stride=1, padding='SAME', data_format=data_format,
                        batch_norm=None, is_training=is_training, activation=None,
                        initializer=initializer, init_factor=init_activation,
                        collection=weight_key)
                with tf.variable_scope('skip_connection{}'.format(l)) as scope:
                    last = tf.add(last, skip2)
                    skip2 = last
            # skip connection
            l += 1
            with tf.variable_scope('conv{}'.format(l)) as scope:
                last = layers.apply_batch_norm(last, decay=batch_norm,
                    is_training=is_training, data_format=data_format)
                last = layers.apply_activation(last, activation=activation,
                    data_format=data_format)
                last = layers.conv2d(last, ksize=3, out_channels=channels,
                    stride=1, padding='SAME', data_format=data_format,
                    batch_norm=None, is_training=is_training, activation=None,
                    initializer=initializer, init_factor=init_activation,
                    collection=weight_key)
            with tf.variable_scope('skip_connection{}'.format(l)) as scope:
                last = tf.add(last, skip1)
                last = layers.apply_activation(last, activation=activation,
                    data_format=data_format)
            # resize conv layer
            l += 1
            with tf.variable_scope('resize_conv{}'.format(l)) as scope:
                last = layers.resize_conv2d(last, ksize=3, out_channels=channels2,
                    scaling=self.scaling, data_format=data_format,
                    batch_norm=None, is_training=is_training, activation=activation,
                    initializer=initializer, init_factor=init_factor,
                    collection=weight_key)
            '''
            # sub-pixel conv layer
            l += 1
            with tf.variable_scope('subpixel_conv{}'.format(l)) as scope:
                last = layers.subpixel_conv2d(last, ksize=3, out_channels=channels2,
                    scaling=self.scaling, data_format=data_format,
                    batch_norm=None, is_training=is_training, activation=activation,
                    initializer=initializer, init_factor=init_factor,
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
            # skip connection
            with tf.variable_scope('skip_connection{}'.format(l)) as scope:
                if data_format == 'NCHW':
                    skip0 = tf.transpose(skip0, (0, 2, 3, 1))
                up_size = tf.shape(skip0)[-3:-1] * self.scaling
                skip0 = tf.image.resize_nearest_neighbor(skip0, up_size)
                if data_format == 'NCHW':
                    skip0 = tf.transpose(skip0, (0, 3, 1, 2))
                last = tf.add(last, skip0)
        # return SR image
        print('Generator: totally {} convolutional layers.'.format(l))
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
        channel_index = -3 if data_format == 'NCHW' else -1
        reduce_strides = [1, 1, 2, 2] if data_format == 'NCHW' else [1, 2, 2, 1]
        # initialization
        last = images
        l = 0
        with tf.variable_scope('discriminator', reuse=reuse) as scope:
            # first conv layer
            l += 1
            with tf.variable_scope('conv{}'.format(l)) as scope:
                last = layers.conv2d(last, ksize=3, out_channels=channels,
                    stride=1, padding='SAME', data_format=data_format,
                    batch_norm=None, is_training=is_training, activation=None,
                    initializer=initializer, init_factor=init_activation,
                    collection=weight_key)
            # residual blocks
            depth = 0
            skip2 = last
            while depth < self.d_depth:
                depth += 1
                reduce_size = True
                strides = reduce_strides if reduce_size else [1, 1, 1, 1]
                double_channel = depth > 1
                if double_channel: channels *= 2
                l += 1
                with tf.variable_scope('conv{}'.format(l)) as scope:
                    last = layers.apply_batch_norm(last, decay=batch_norm,
                        is_training=is_training, data_format=data_format)
                    last = layers.apply_activation(last, activation=activation,
                        data_format=data_format)
                    last = layers.conv2d(last, ksize=2, out_channels=channels,
                        stride=strides, padding='SAME', data_format=data_format,
                        batch_norm=None, is_training=is_training, activation=None,
                        initializer=initializer, init_factor=init_activation,
                        collection=weight_key)
                l += 1
                with tf.variable_scope('conv{}'.format(l)) as scope:
                    last = layers.apply_batch_norm(last, decay=batch_norm,
                        is_training=is_training, data_format=data_format)
                    last = layers.apply_activation(last, activation=activation,
                        data_format=data_format)
                    last = layers.conv2d(last, ksize=3, out_channels=channels,
                        stride=1, padding='SAME', data_format=data_format,
                        batch_norm=None, is_training=is_training, activation=None,
                        initializer=initializer, init_factor=init_activation,
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
            # final conv layer
            channels *= 2
            l += 1
            with tf.variable_scope('conv{}'.format(l)) as scope:
                last = layers.conv2d(last, ksize=2, out_channels=channels,
                    stride=2, padding='SAME', data_format=data_format,
                    batch_norm=None, is_training=is_training, activation=activation,
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
        print('Critic: totally {} convolutional/dense layers.'.format(l))
        return last
    
    def generator_losses(self, ref, pred, pred_logits, alpha=0.0, weights1=1.0, weights2=1.0):
        import utils.image
        collection = self.generator_loss_key
        with tf.variable_scope('generator_losses') as scope:
            # adversarial loss
            if self.gan_loss == 1:
                g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=pred_logits, labels=tf.ones_like(pred_logits)))
                tf.summary.scalar('g_loss', g_loss)
                ad_loss = tf.multiply(g_loss, 1e-2, name='adversarial_loss')
            elif self.gan_loss in [2, 3, 4, 5]:
                g_loss = tf.reduce_mean(pred_logits, name='g_loss')
                ad_loss = tf.multiply(g_loss, -1e-3, name='adversarial_loss')
            tf.losses.add_loss(ad_loss, loss_collection=collection)
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
    
    def discriminator_losses(self, ref_logits, pred_logits, ref, pred):
        collection = self.discriminator_loss_key
        # WGAN lipschitz-penalty
        def random_interpolate():
            alpha = tf.random_uniform(shape=tf.shape(ref), minval=0., maxval=1.)
            differences = pred - ref
            interpolates = alpha * differences + ref
            return interpolates
        if self.gan_loss == 2:
            # compute gradients directly
            # https://github.com/igul222/improved_wgan_training
            inter = random_interpolate()
            inter_logits = self.discriminator(inter, is_training=True, reuse=True)
            gradients = tf.gradients(inter_logits, [inter])[0]
            slopes = tf.norm(tf.contrib.layers.flatten(gradients), axis=1)
        elif self.gan_loss == 3:
            # compute gradients using Hessian-Vector products trick
            # https://justindomke.wordpress.com/2009/01/17/hessian-vector-products/
            small_r = 1e-6
            alpha = tf.random_uniform(shape=tf.shape(ref), minval=0., maxval=1.)
            differences = pred - ref
            inter1 = (alpha + small_r) * differences + ref
            inter2 = (alpha - small_r) * differences + ref
            x_diff = (2 * small_r) * differences
            inter_logits1 = self.discriminator(inter1, is_training=True, reuse=True)
            inter_logits2 = self.discriminator(inter2, is_training=True, reuse=True)
            y_diff = tf.abs(inter_logits1 - inter_logits2)
            slopes = y_diff / tf.norm(tf.contrib.layers.flatten(x_diff), axis=1)
        elif self.gan_loss == 4:
            # compute slopes using finite difference of 2 random samples
            # https://www.zhihu.com/question/52602529/answer/158727900
            inter1 = random_interpolate()
            inter2 = random_interpolate()
            x_diff = inter1 - inter2
            inter_logits1 = self.discriminator(inter1, is_training=True, reuse=True)
            inter_logits2 = self.discriminator(inter2, is_training=True, reuse=True)
            y_diff = tf.abs(inter_logits1 - inter_logits2)
            slopes = y_diff / tf.norm(tf.contrib.layers.flatten(x_diff), axis=1)
        elif self.gan_loss == 5:
            # compute slopes using difference of real and fake samples
            x_diff = pred - ref
            y_diff = tf.abs(pred_logits - ref_logits)
            slopes = y_diff / tf.norm(tf.contrib.layers.flatten(x_diff), axis=1)
        with tf.variable_scope('discriminator_losses') as scope:
            # L2 regularization weight decay
            if self.d_weight_decay > 0:
                with tf.variable_scope('l2_regularize') as scope:
                    l2_regularize = tf.add_n([tf.nn.l2_loss(v) for v in
                        tf.get_collection(self.discriminator_weight_key)])
                    l2_regularize = tf.multiply(l2_regularize, self.d_weight_decay, name='loss')
                    tf.losses.add_loss(l2_regularize, loss_collection=collection)
            # adversarial loss
            if self.gan_loss == 1:
                d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=ref_logits, labels=tf.ones_like(ref_logits)))
                d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=pred_logits, labels=tf.zeros_like(pred_logits)))
                tf.summary.scalar('d_loss_real', d_loss_real)
                tf.summary.scalar('d_loss_fake', d_loss_fake)
                ad_loss = tf.add(d_loss_real, d_loss_fake, name='adversarial_loss')
            elif self.gan_loss in [2, 3, 4, 5]:
                d_real = tf.reduce_mean(ref_logits)
                d_fake = tf.reduce_mean(pred_logits)
                d_loss = d_fake - d_real
                # WGAN lipschitz-penalty
                K = 1.
                gradient_penalty = tf.reduce_mean(tf.square(slopes - K))
                lp_loss = self.lp_lambda * gradient_penalty
                # summary
                tf.summary.scalar('d_real', d_real)
                tf.summary.scalar('d_fake', d_fake)
                tf.summary.scalar('lp_loss', lp_loss)
                ad_loss = tf.add(d_loss, lp_loss, name='adversarial_loss')
            tf.losses.add_loss(ad_loss, loss_collection=collection)
            # return total loss
            return tf.add_n(tf.losses.get_losses(loss_collection=collection),
                            name=self.discriminator_total_loss_key)
    
    def build_model(self, images_lr=None, is_training=False):
        # set inputs
        if images_lr is None:
            self.images_lr = tf.placeholder(self.dtype, self.shape_lr, name='Input')
        else:
            self.images_lr = tf.identity(images_lr, name='Input')
            self.images_lr.set_shape(self.shape_lr)
        if self.input_range == 2:
            self.images_lr = self.images_lr * 2 - 1
        
        # apply generator to inputs
        self.images_sr = self.generator(self.images_lr, is_training=is_training)
        
        # generated outputs
        self.images_sr.set_shape(self.shape_hr)
        
        # restore [0, 1] range for generated outputs
        if self.output_range == 2:
            tf.multiply(self.images_sr + 1, 0.5, name='Output')
        else:
            tf.identity(self.images_sr, name='Output')
        
        # trainable and model variables
        self.g_tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        self.g_mvars = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope='generator')
        self.g_svars = list(set(self.g_tvars + self.g_mvars))
        
        # track the moving averages of all trainable variables
        if not is_training and self.train_moving_average > 0:
            ema = tf.train.ExponentialMovingAverage(self.train_moving_average,
                name='train_moving_average')
            g_ema_op = ema.apply(self.g_svars)
            self.g_svars = {ema.average_name(var): var for var in self.g_svars}
        
        # return generated results
        return self.images_sr
    
    def build_train(self, images_lr=None, images_hr=None):
        # reference outputs - from data generating distribution
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
        '''
        d_logits_hr = self.discriminator(self.images_hr, is_training=True)
        d_logits_sr = self.discriminator(self.images_sr, is_training=True, reuse=True)
        '''
        images = tf.concat([self.images_hr, self.images_sr], axis=0)
        d_logits = self.discriminator(images, is_training=True)
        d_logits_hr, d_logits_sr = tf.split(d_logits, 2, axis=0)
        
        # build generator losses
        self.g_loss = self.generator_losses(self.images_hr, self.images_sr, d_logits_sr)
        
        # build discriminator losses
        self.d_loss = self.discriminator_losses(d_logits_hr, d_logits_sr,
                                                self.images_hr, self.images_sr)
        
        # trainable and model variables
        self.d_tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        self.d_mvars = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope='discriminator')
        self.d_svars = list(set(self.d_tvars + self.d_mvars))
        
        # return total loss(es)
        return self.g_loss, self.d_loss
    
    def train(self, global_step):
        print('lr: {}|{}, decay steps: {}, decay factor: {}, min: {}|{}, weight decay: {}|{}'.format(
            self.learning_rate, self.d_learning_rate, self.lr_decay_steps, self.lr_decay_factor,
            self.lr_min, self.d_lr_min, self.weight_decay, self.d_weight_decay))
        
        # training ops
        g_train_ops = []
        d_train_ops = []
        
        # dependency need to be updated
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        
        # generate moving averages of all losses and associated summaries
        losses = [self.g_loss, self.d_loss]
        losses += tf.losses.get_losses(loss_collection=self.generator_loss_key)
        losses += tf.losses.get_losses(loss_collection=self.discriminator_loss_key)
        loss_averages_op = layers.loss_summaries(losses, self.loss_moving_average)
        if loss_averages_op: update_ops.append(loss_averages_op)
        
        # decay the learning rate exponentially based on the number of steps
        if self.lr_decay_steps > 0 and self.lr_decay_factor != 0:
            g_lr = tf.train.exponential_decay(self.learning_rate, global_step,
                self.lr_decay_steps, 1 - self.lr_decay_factor, staircase=True)
            if self.lr_min > 0:
                g_lr = tf.maximum(tf.constant(self.lr_min, dtype=self.dtype), g_lr)
            d_lr = tf.train.exponential_decay(self.d_learning_rate, global_step,
                self.lr_decay_steps, 1 - self.lr_decay_factor, staircase=True)
            if self.d_lr_min > 0:
                g_lr = tf.maximum(tf.constant(self.d_lr_min, dtype=self.dtype), d_lr)
        else:
            g_lr = self.learning_rate
            d_lr = self.d_learning_rate
        tf.summary.scalar('g_learning_rate', g_lr)
        tf.summary.scalar('d_learning_rate', d_lr)
        
        # optimizer
        g_opt = tf.train.NadamOptimizer(g_lr, beta1=self.learning_beta1,
            beta2=self.learning_beta2, epsilon=self.epsilon)
        d_opt = tf.train.NadamOptimizer(d_lr, beta1=self.learning_beta1,
            beta2=self.learning_beta2, epsilon=self.epsilon)
        
        # compute gradients
        with tf.control_dependencies(update_ops):
            g_grads_and_vars = g_opt.compute_gradients(self.g_loss, self.g_tvars)
            d_grads_and_vars = d_opt.compute_gradients(self.d_loss, self.d_tvars)
        
        # apply gradient
        g_train_ops.append(g_opt.apply_gradients(g_grads_and_vars, global_step))
        d_train_ops.append(d_opt.apply_gradients(d_grads_and_vars, None))
        
        # add histograms for gradients
        for grad, var in g_grads_and_vars + d_grads_and_vars:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)
            if var is not None:
                tf.summary.histogram(var.op.name, var)
        
        # track the moving averages of all trainable variables
        if self.train_moving_average > 0:
            ema = tf.train.ExponentialMovingAverage(self.train_moving_average,
                global_step, name='train_moving_average')
            g_ema_op = ema.apply(self.g_svars)
            g_train_ops.append(g_ema_op)
            self.g_svars = [ema.average(var) for var in self.g_svars]
        
        # generate operation
        with tf.control_dependencies(g_train_ops):
            g_train_op = tf.no_op(name='g_train')
        with tf.control_dependencies(d_train_ops):
            d_train_op = tf.no_op(name='d_train')
        return g_train_op, d_train_op
