import sys
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
sys.path.append('..')
from utils import layers

# flags
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('graph_dir', './graph.tmp',
                           """Directory where to write meta graph and data.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")
tf.app.flags.DEFINE_string('device', 'GPU:0',
                            """Preferred device to use.""")
tf.app.flags.DEFINE_string('data_format', 'NCHW',
                            """Data layout format.""")
tf.app.flags.DEFINE_integer('patch_height', 1024,
                            """Block size y.""")
tf.app.flags.DEFINE_integer('patch_width', 1024,
                            """Block size x.""")
tf.app.flags.DEFINE_integer('batch_size', 1,
                            """Batch size.""")
tf.app.flags.DEFINE_integer('res_blocks', 6,
                            """Number of residual blocks.""")
tf.app.flags.DEFINE_integer('channels', 64,
                            """Number of features in hidden layers.""")
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

def ResNet_3_3(last):
    is_training = False
    channels = FLAGS.channels
    l = 0
    last = tf.identity(last, name='input')
    # residual blocks
    rb = 0
    skip2 = last
    while rb < FLAGS.res_blocks:
        rb += 1
        l += 1
        with tf.variable_scope('conv{}'.format(l)) as scope:
            last = layers.conv2d(last, ksize=3, out_channels=channels,
                                 stride=1, padding='SAME', data_format=FLAGS.data_format,
                                 bn=FLAGS.batch_norm, train=is_training, activation=FLAGS.activation,
                                 init_factor=FLAGS.init_activation)
        l += 1
        with tf.variable_scope('conv{}'.format(l)) as scope:
            last = layers.conv2d(last, ksize=3, out_channels=channels,
                                 stride=1, padding='SAME', data_format=FLAGS.data_format,
                                 bn=FLAGS.batch_norm, train=is_training, activation=None,
                                 init_factor=FLAGS.init_factor)
        with tf.variable_scope('skip_connection{}'.format(l)) as scope:
            last = tf.add(last, skip2, 'elementwise_sum')
            skip2 = last
            last = layers.apply_activation(last, activation=FLAGS.activation,
                                           data_format=FLAGS.data_format)
    # return
    last = tf.identity(last, name='output')
    return last

def ResNet_1_3_1(last):
    is_training = False
    channels = FLAGS.channels
    l = 0
    last = tf.identity(last, name='input')
    # residual blocks
    rb = 0
    skip2 = last
    while rb < FLAGS.res_blocks:
        rb += 1
        l += 1
        with tf.variable_scope('conv{}'.format(l)) as scope:
            last = layers.conv2d(last, ksize=1, out_channels=channels,
                                 stride=1, padding='SAME', data_format=FLAGS.data_format,
                                 bn=FLAGS.batch_norm, train=is_training, activation=FLAGS.activation,
                                 init_factor=FLAGS.init_activation)
        l += 1
        with tf.variable_scope('conv{}'.format(l)) as scope:
            last = layers.conv2d(last, ksize=3, out_channels=channels,
                                 stride=1, padding='SAME', data_format=FLAGS.data_format,
                                 bn=FLAGS.batch_norm, train=is_training, activation=FLAGS.activation,
                                 init_factor=FLAGS.init_activation)
        l += 1
        with tf.variable_scope('conv{}'.format(l)) as scope:
            last = layers.conv2d(last, ksize=1, out_channels=channels,
                                 stride=1, padding='SAME', data_format=FLAGS.data_format,
                                 bn=FLAGS.batch_norm, train=is_training, activation=None,
                                 init_factor=FLAGS.init_factor)
        with tf.variable_scope('skip_connection{}'.format(l)) as scope:
            last = tf.add(last, skip2, 'elementwise_sum')
            skip2 = last
            last = layers.apply_activation(last, activation=FLAGS.activation,
                                           data_format=FLAGS.data_format)
    # return
    last = tf.identity(last, name='output')
    return last

def Xception(last):
    is_training = False
    channels = FLAGS.channels
    l = 0
    last = tf.identity(last, name='input')
    # residual blocks
    rb = 0
    skip2 = last
    while rb < FLAGS.res_blocks:
        rb += 1
        l += 1
        with tf.variable_scope('separable_conv{}'.format(l)) as scope:
            last = layers.separable_conv2d(last, ksize=3, channel_multiplier=1, out_channels=channels,
                                           stride=1, padding='SAME', data_format=FLAGS.data_format,
                                           bn=FLAGS.batch_norm, train=is_training, activation=FLAGS.activation,
                                           init_factor=FLAGS.init_activation)
        l += 1
        with tf.variable_scope('separable_conv{}'.format(l)) as scope:
            last = layers.separable_conv2d(last, ksize=3, channel_multiplier=1, out_channels=channels,
                                           stride=1, padding='SAME', data_format=FLAGS.data_format,
                                           bn=FLAGS.batch_norm, train=is_training, activation=None,
                                           init_factor=FLAGS.init_activation)
        with tf.variable_scope('skip_connection{}'.format(l)) as scope:
            last = tf.add(last, skip2, 'elementwise_sum')
            skip2 = last
            last = layers.apply_activation(last, activation=FLAGS.activation)
    # return
    last = tf.identity(last, name='output')
    return last

def ResNeXt(last, group_num=16):
    is_training = False
    channels = FLAGS.channels
    channel_index = -3 if FLAGS.data_format == 'NCHW' else -1
    l = 0
    last = tf.identity(last, name='input')
    # residual blocks
    rb = 0
    skip2 = last
    while rb < FLAGS.res_blocks:
        rb += 1
        l += 1
        with tf.variable_scope('conv{}'.format(l)) as scope:
            last = layers.conv2d(last, ksize=1, out_channels=channels,
                                 stride=1, padding='SAME', data_format=FLAGS.data_format,
                                 bn=FLAGS.batch_norm, train=is_training, activation=FLAGS.activation,
                                 init_factor=FLAGS.init_activation)
        '''
        # original version - individual weights for each group
        l += 1
        with tf.variable_scope('group_conv{}'.format(l)) as scope:
            group = tf.split(last, group_num, axis=channel_index)
            for _ in range(group_num):
                with tf.variable_scope('group{}'.format(_ + 1)) as scope:
                    group[_] = layers.conv2d(group[_], ksize=3, out_channels=channels // group_num,
                                             stride=1, padding='SAME', data_format=FLAGS.data_format,
                                             bn=FLAGS.batch_norm, train=is_training, activation=None,
                                             init_factor=FLAGS.init_activation)
            last = tf.concat(group, axis=channel_index)
            last = layers.apply_activation(last, activation=FLAGS.activation,
                                           data_format=FLAGS.data_format)
        '''
        # modified version - shared weights between groups
        l += 1
        with tf.variable_scope('group_conv{}'.format(l)) as scope:
            # channel en-batching
            shape = tf.shape(last)
            if FLAGS.data_format == 'NCHW':
                shape_batch = [shape[0] * group_num, channels // group_num, shape[2], shape[3]]
            else:
                shape_divide = [shape[0], shape[1], shape[2], group_num, channels // group_num]
                last = tf.reshape(last, shape_divide)
                last = tf.transpose(last, (0, 3, 1, 2, 4))
                shape_batch = [shape[0] * group_num, shape[1], shape[2], channels // group_num]
            last = tf.reshape(last, shape_batch)
            # convolution
            last = layers.conv2d(last, ksize=3, out_channels=channels // group_num,
                                 stride=1, padding='SAME', data_format=FLAGS.data_format,
                                 bn=FLAGS.batch_norm, train=is_training, activation=None,
                                 init_factor=FLAGS.init_activation)
            # channel de-batching
            if FLAGS.data_format != 'NCHW':
                shape_divide = [shape[0], group_num, shape[1], shape[2], channels // group_num]
                last = tf.reshape(last, shape_divide)
                last = tf.transpose(last, (0, 2, 3, 1, 4))
            last = tf.reshape(last, shape)
            # activation
            last = layers.apply_activation(last, activation=FLAGS.activation,
                                           data_format=FLAGS.data_format)
        l += 1
        with tf.variable_scope('conv{}'.format(l)) as scope:
            last = layers.conv2d(last, ksize=1, out_channels=channels,
                                 stride=1, padding='SAME', data_format=FLAGS.data_format,
                                 bn=FLAGS.batch_norm, train=is_training, activation=None,
                                 init_factor=FLAGS.init_factor)
        with tf.variable_scope('skip_connection{}'.format(l)) as scope:
            last = tf.add(last, skip2, 'elementwise_sum')
            skip2 = last
            last = layers.apply_activation(last, activation=FLAGS.activation,
                                           data_format=FLAGS.data_format)
    # return
    last = tf.identity(last, name='output')
    return last

def ShuffleNet(last, group_num=8):
    is_training = False
    channels = FLAGS.channels
    channel_index = -3 if FLAGS.data_format == 'NCHW' else -1
    l = 0
    last = tf.identity(last, name='input')
    # residual blocks
    rb = 0
    skip2 = last
    while rb < FLAGS.res_blocks:
        rb += 1
        l += 1
        with tf.variable_scope('pointwise_group_conv{}'.format(l)) as scope:
            group = tf.split(last, group_num, axis=channel_index)
            for _ in range(group_num):
                with tf.variable_scope('group{}'.format(_ + 1)) as scope:
                    group[_] = layers.conv2d(group[_], ksize=1, out_channels=channels // group_num,
                                             stride=1, padding='SAME', data_format=FLAGS.data_format,
                                             bn=FLAGS.batch_norm, train=is_training, activation=None,
                                             init_factor=FLAGS.init_activation)
            last = tf.concat(group, axis=channel_index)
            last = layers.apply_activation(last, activation=FLAGS.activation,
                                           data_format=FLAGS.data_format)
        with tf.variable_scope('channel_shuffle{}'.format(l)) as scope:
            shape = tf.shape(last)
            if FLAGS.data_format == 'NCHW':
                shape_divide = [shape[0], group_num, channels // group_num, shape[2], shape[3]]
            else:
                shape_divide = [shape[0], shape[1], shape[2], group_num, channels // group_num]
            last = tf.reshape(last, shape_divide)
            trans = (0, 2, 1, 3, 4) if FLAGS.data_format == 'NCHW' else (0, 1, 2, 4, 3)
            last = tf.transpose(last, trans)
            last = tf.reshape(last, shape)
        l += 1
        with tf.variable_scope('depthwise_conv{}'.format(l)) as scope:
            last = layers.depthwise_conv2d(last, ksize=3, channel_multiplier=1,
                                           stride=1, padding='SAME', data_format=FLAGS.data_format,
                                           bn=FLAGS.batch_norm, train=is_training, activation=None,
                                           init_factor=FLAGS.init_activation)
        l += 1
        with tf.variable_scope('pointwise_group_conv{}'.format(l)) as scope:
            group = tf.split(last, group_num, axis=channel_index)
            for _ in range(group_num):
                with tf.variable_scope('group{}'.format(_ + 1)) as scope:
                    group[_] = layers.conv2d(group[_], ksize=1, out_channels=channels // group_num,
                                             stride=1, padding='SAME', data_format=FLAGS.data_format,
                                             bn=FLAGS.batch_norm, train=is_training, activation=None,
                                             init_factor=FLAGS.init_activation)
            last = tf.concat(group, axis=channel_index)
        with tf.variable_scope('skip_connection{}'.format(l)) as scope:
            last = tf.add(last, skip2, 'elementwise_sum')
            skip2 = last
            last = layers.apply_activation(last, activation=FLAGS.activation)
    # return
    last = tf.identity(last, name='output')
    return last

# setup tensorflow and return session
def setup():
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options,
        allow_soft_placement=True, log_device_placement=False)
    sess = tf.Session(config=config)
    return sess

def bench():
    # initialization
    channels = FLAGS.channels
    if FLAGS.data_format == 'NCHW':
        shape = (FLAGS.batch_size, channels, FLAGS.patch_height, FLAGS.patch_width)
    else:
        shape = (FLAGS.batch_size, FLAGS.patch_height, FLAGS.patch_width, channels)

    # inferences
    graphs = []
    def add_res(func):
        g = tf.Graph()
        graphs.append(g)
        with g.as_default():
            with g.device('/device:{}'.format(FLAGS.device)):
                data = tf.placeholder(tf.float32, shape)
                func(data)
    
    add_res(lambda data: ResNet_3_3(data))
    add_res(lambda data: ResNet_1_3_1(data))
    add_res(lambda data: Xception(data))
    add_res(lambda data: ResNeXt(data, 2))
    add_res(lambda data: ResNeXt(data, 4))
    add_res(lambda data: ResNeXt(data, 8))
    add_res(lambda data: ResNeXt(data, 16))
    add_res(lambda data: ShuffleNet(data, 2))
    add_res(lambda data: ShuffleNet(data, 4))
    add_res(lambda data: ShuffleNet(data, 8))

    # session
    loop = 10 if FLAGS.device[:3].lower() == 'cpu' else 100
    next_data = np.empty(shape, dtype=np.float32)
    feed_dict = {'input:0': next_data}
    
    num = 0
    for g in graphs:
        num += 1
        with g.as_default():
            output = g.get_tensor_by_name('output:0')
            with setup() as sess:
                # initialize variables
                sess.run(tf.global_variables_initializer())
                # save the graph
                saver = tf.train.Saver()
                saver.save(sess, os.path.join(FLAGS.graph_dir, 'model{}'.format(num)),
                           write_meta_graph=True, write_state=False)
                # warm-up
                sess.run(output, feed_dict)
                # run options
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                # benchmark
                start = time.time()
                for _ in range(loop):
                    if _ == loop - 1:
                        sess.run(output, feed_dict, options=run_options, run_metadata=run_metadata)
                        # Create the Timeline object, and write it to a json
                        tl = timeline.Timeline(run_metadata.step_stats)
                        ctf = tl.generate_chrome_trace_format()
                        with open(os.path.join(FLAGS.graph_dir, 'timeline_{:0>2}.json'.format(num)), 'a') as f:
                            f.write(ctf)
                    else:
                        sess.run(output, feed_dict)
                duration = time.time() - start
                # print
                print('Time for {} loops: {} s'.format(loop, duration))

def main(argv=None):
    # arXiv 1509.09308
    # a new class of fast algorithms for convolutional neural networks using Winograd's minimal filtering algorithms
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    
    if tf.gfile.Exists(FLAGS.graph_dir):
        tf.gfile.DeleteRecursively(FLAGS.graph_dir)
    tf.gfile.MakeDirs(FLAGS.graph_dir)
    bench()

if __name__ == '__main__':
    tf.app.run()

'''
Time (s)
========
--batch_size 16 --patch_height 96 --patch_width 96
                GPU (100)   CPU (10)
--------
ResNet_3_3      13.546      31.151
ResNet_1_3_1    16.922      26.575
Xception        14.425      14.063
ResNeXt(2)      18.132      24.395
ResNeXt(4)      18.189      24.369
ResNeXt(8)      19.217      24.098
ResNeXt(16)     20.085      24.644
ShuffleNet(2)   17.577      21.565
ShuffleNet(4)   17.779      21.513
ShuffleNet(8)   18.575      22.522
========
ResNeXt: modified group conv
--batch_size 16 --patch_height 96 --patch_width 96
--------
ResNet_3_3      14.045
ResNeXt(2)      17.050
ResNeXt(4)      17.166
ResNeXt(8)      18.138
ResNeXt(16)     18.762
========
--batch_size 1 --patch_height 512 --patch_width 512
ResNeXt: modified group conv
TF_ENABLE_WINOGRAD_NONFUSED=0
--------
ResNet_3_3      23.698
ResNet_1_3_1    29.273
Xception        27.497
ResNeXt(2)      28.472
ResNeXt(4)      28.673
ResNeXt(8)      30.228
ResNeXt(16)     31.511
ShuffleNet(2)   33.593
ShuffleNet(4)   34.087
ShuffleNet(8)   35.334
========
--batch_size 1 --patch_height 512 --patch_width 512
ResNeXt: modified group conv
TF_ENABLE_WINOGRAD_NONFUSED=1
--------
ResNet_3_3      22.863
ResNet_1_3_1    28.595
Xception        26.739
ResNeXt(2)      28.016
ResNeXt(4)      28.048
ResNeXt(8)      29.438
ResNeXt(16)     30.738
ShuffleNet(2)   32.481
ShuffleNet(4)   32.717
ShuffleNet(8)   34.119
========
GPU (100)
res_blocks=6
channels        32      48      64      80      96      112
--------
SpeedRatioTo64  .52445  .73157  1.0000  1.1268  1.3175  1.3161
--------
ResNet_3_3      6.7304  10.856  14.119  19.578  24.113  32.853
Xception        8.3885  12.464  16.471  21.718  25.992  29.381
ResNeXt(2)      9.6206  14.037  18.692  25.397  30.102  33.969
ShuffleNet(2)   10.183  15.040  19.802  24.868  29.646  36.570
========
GPU (100)
channels=64
res_blocks      2       4       6       8       10      12      14    
--------
SpeedRatioTo8   .83584  .93530  .98030  1.0000  1.0132  1.0227  1.0284
--------
ResNet_3_3      5.4963  9.8236  14.059  18.376  22.670  26.951  31.269
Xception        6.2584  11.363  16.464  21.590  26.657  31.760  36.788
ResNeXt(2)      7.0044  12.841  18.637  24.421  30.239  36.075  41.772
ShuffleNet(2)   7.4054  13.629  20.052  26.066  32.198  38.426  44.935
========
GPU (100)
batch           16x96x96    32x96x96    16x128x128  8x192x192   4x256x256   1x512x512
--------
ResNet_3_3      14.080      27.862      24.749      27.920      24.808      23.782
Xception        16.498      32.566      30.197      31.352      29.005      27.471
ResNeXt(2)      18.654      36.889      32.592      36.977      32.979      31.093
ShuffleNet(2)   19.869      39.288      35.587      38.762      35.425      33.671
'''
