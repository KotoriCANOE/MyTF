import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
sys.path.append('..')
from utils import helper
import utils.image

from input import inputs
import model

# working directory
print('Current working directory:\n    {}\n'.format(os.getcwd()))

# flags
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('postfix', '',
                            """Postfix added to train_dir, test_dir, test files, etc.""")
tf.app.flags.DEFINE_string('train_dir', './train{}.tmp'.format(FLAGS.postfix),
                           """Directory where to read checkpoint.""")
tf.app.flags.DEFINE_string('test_dir', './test{}.tmp'.format(FLAGS.postfix),
                           """Directory where to write event logs and test results.""")
tf.app.flags.DEFINE_integer('random_seed', 0,
                            """Initialize with specified random seed.""")
tf.app.flags.DEFINE_integer('threads', 8,
                            """Number of threads for Dataset process.""")
tf.app.flags.DEFINE_integer('threads_py', 4,
                            """Number of threads for Dataset process in tf.py_func.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_string('data_format', 'NCHW', # 'NHWC'
                            """Data layout format.""")
tf.app.flags.DEFINE_integer('patch_height', 512,
                            """Block size y.""")
tf.app.flags.DEFINE_integer('patch_width', 512,
                            """Block size x.""")
tf.app.flags.DEFINE_integer('batch_size', 1,
                            """Batch size.""")
tf.app.flags.DEFINE_boolean('pre_down', True,
                            """Pre-downscale large image for (probably) higher quality data.""")
tf.app.flags.DEFINE_float('noise_scale', 0.01,
                            """STD of additive Gaussian random noise.""")
tf.app.flags.DEFINE_float('noise_corr', 0.75,
                            """Spatial correlation of the Gaussian random noise.""")
tf.app.flags.DEFINE_boolean('jpeg_coding', True,
                            """Using JPEG to generate compression artifacts for data.""")

# constants
TESTSET_PATH = r'..\Dataset.SR\Test'

# setup tensorflow
def setup():
    # create session
    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # initialize rng with a deterministic seed
    import random
    with sess.graph.as_default():
        tf.set_random_seed(FLAGS.random_seed)
    random.seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)

    summary_writer = tf.summary.FileWriter(FLAGS.test_dir, sess.graph)
    return sess, summary_writer

# losses
def get_losses(gtruth, pred):
    # RGB color space
    RGB_mse = tf.losses.mean_squared_error(gtruth, pred, weights=1.0)
    RGB_mad = tf.losses.absolute_difference(gtruth, pred, weights=1.0)
    # OPP color space - Y
    Y_gtruth = utils.image.RGB2Y(gtruth, data_format=FLAGS.data_format)
    Y_pred = utils.image.RGB2Y(pred, data_format=FLAGS.data_format)
    Y_ss_ssim = utils.image.SS_SSIM(Y_gtruth, Y_pred, data_format=FLAGS.data_format)
    Y_ms_ssim = utils.image.MS_SSIM2(Y_gtruth, Y_pred, norm=True, data_format=FLAGS.data_format)
    return RGB_mse, RGB_mad, Y_ss_ssim, Y_ms_ssim

def total_loss():
    return tf.losses.get_total_loss()

# testing
def test():
    files = helper.listdir_files(TESTSET_PATH,
                                 filter_ext=['.jpeg', '.jpg', '.png'],
                                 encoding=True)
    steps_per_epoch = len(files) // FLAGS.batch_size
    epoch_size = steps_per_epoch * FLAGS.batch_size
    max_steps = steps_per_epoch
    files = files[:epoch_size]
    
    with tf.Graph().as_default():
        # setup global tensorflow state
        sess, summary_writer = setup()
        
        # pre-processing for input
        with tf.device('/cpu:0'):
            images_lr, images_hr = inputs(files, is_testing=True)
        
        # model inference and losses
        images_sr = model.inference(images_lr, is_training=False)
        ret_loss = list(get_losses(images_hr, images_sr))
        
        # restore variables from checkpoint
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.train_dir))
        
        # post-processing for output
        with tf.device('/cpu:0'):
            # data format conversion
            if FLAGS.data_format == 'NCHW':
                images_lr = utils.image.NCHW2NHWC(images_lr)
                images_hr = utils.image.NCHW2NHWC(images_hr)
                images_sr = utils.image.NCHW2NHWC(images_sr)
            
            # Bicubic upscaling
            shape = tf.shape(images_lr)
            upsize = [shape[-3] * FLAGS.scaling, shape[-2] * FLAGS.scaling]
            images_bicubic = tf.image.resize_images(images_lr, upsize,
                    tf.image.ResizeMethod.BICUBIC, align_corners=True)
            
            # PNGs output
            ret_pngs = []
            ret_pngs.extend(helper.BatchPNG(images_lr, FLAGS.batch_size))
            ret_pngs.extend(helper.BatchPNG(images_hr, FLAGS.batch_size))
            ret_pngs.extend(helper.BatchPNG(images_sr, FLAGS.batch_size))
            ret_pngs.extend(helper.BatchPNG(images_bicubic, FLAGS.batch_size))
        
        # options
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        # run session
        ret = ret_loss + ret_pngs
        sum_loss = [0 for _ in range(len(ret_loss))]
        for step in range(max_steps):
            '''
            if step % 20 == 0:
                cur_ret = sess.run(ret, options=run_options, run_metadata=run_metadata)
                # Create the Timeline object, and write it to a json
                tl = timeline.Timeline(run_metadata.step_stats)
                ctf = tl.generate_chrome_trace_format()
                with open(os.path.join(FLAGS.test_dir, 'timeline_{:0>7}.json'.format(step)), 'a') as f:
                    f.write(ctf)
            else:
                cur_ret = sess.run(ret)
            '''
            cur_ret = sess.run(ret)
            cur_loss = cur_ret[0:len(ret_loss)]
            cur_pngs = cur_ret[len(ret_loss):]
            # monitor losses
            for _ in range(len(ret_loss)):
                sum_loss[_] += cur_loss[_]
            print('batch {}, MSE (RGB) {}, MAD (RGB) {}, SS-SSIM(Y) {}, MS-SSIM (Y) {}'.format(
                   step, *cur_loss))
            # images output
            _start = step * FLAGS.batch_size
            _stop = _start + FLAGS.batch_size
            _range = range(_start, _stop)
            ofiles = []
            ofiles.extend([os.path.join(FLAGS.test_dir,
                '{:0>5}.0.LR.png'.format(_)) for _ in _range])
            ofiles.extend([os.path.join(FLAGS.test_dir,
                '{:0>5}.1.HR.png'.format(_)) for _ in _range])
            ofiles.extend([os.path.join(FLAGS.test_dir,
                '{:0>5}.2.SR{}.png'.format(_, FLAGS.postfix)) for _ in _range])
            ofiles.extend([os.path.join(FLAGS.test_dir,
                '{:0>5}.3.Bicubic.png'.format(_)) for _ in _range])
            helper.WriteFiles(cur_pngs, ofiles)
        sess.close()
        
        # summary
        mean_loss = [l / max_steps for l in sum_loss]
        psnr = 10 * np.log10(1 / mean_loss[0]) if mean_loss[0] > 0 else 100
        print('PSNR (RGB) {}, MAD (RGB) {}, SS-SSIM(Y) {}, MS-SSIM (Y) {}'.format(
               psnr, *mean_loss[1:]))

# main
def main(argv=None):
    if not tf.gfile.IsDirectory(FLAGS.train_dir):
        raise FileNotFoundError('Could not find folder {}'.format(FLAGS.train_dir))
    if tf.gfile.Exists(FLAGS.test_dir):
        tf.gfile.DeleteRecursively(FLAGS.test_dir)
    tf.gfile.MakeDirs(FLAGS.test_dir)
    test()

if __name__ == '__main__':
    tf.app.run()
