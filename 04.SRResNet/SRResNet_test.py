import sys
import os
import numpy as np
import tensorflow as tf
sys.path.append('..')
from utils import helper
import utils.image

from SRResNet_input import inputs
import SRResNet as model

# working directory
print('Current working directory:\n    {}\n'.format(os.getcwd()))

# flags
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', './train.tmp',
                           """Directory where to read checkpoint.""")
tf.app.flags.DEFINE_string('test_dir', './test.tmp',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_integer('random_seed', 0,
                            """Initialize with specified random seed.""")
tf.app.flags.DEFINE_integer('threads', 4,
                            """Number of threads for Dataset process.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('patch_height', 512,
                            """Block size y.""")
tf.app.flags.DEFINE_integer('patch_width', 512,
                            """Block size x.""")
tf.app.flags.DEFINE_integer('batch_size', 1,
                            """Batch size.""")

# constants
TESTSET_PATH = r'..\Dataset.SR\Test'

# setup tensorflow
def setup():
    # create session
    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # initialize rng with a deterministic seed
    with sess.graph.as_default():
        tf.set_random_seed(FLAGS.random_seed)
    #random.seed(FLAGS.random_seed)
    #np.random.seed(FLAGS.random_seed)

    summary_writer = tf.summary.FileWriter(FLAGS.test_dir, sess.graph)
    return sess, summary_writer

# losses
def get_losses(images_hr, images_sr):
    # RGB loss
    RGB_mse = tf.losses.mean_squared_error(images_hr, images_sr, weights=1.0)
    RGB_mad = tf.losses.absolute_difference(images_hr, images_sr, weights=1.0)
    # OPP loss
    images_hr = utils.image.RGB2OPP(images_hr, norm=False)
    images_sr = utils.image.RGB2OPP(images_sr, norm=False)
    OPP_mad = tf.losses.absolute_difference(images_hr, images_sr, weights=1.0)
    Y_msssim = utils.image.MS_SSIM(images_hr[:,:,:,:1], images_sr[:,:,:,:1])
    return RGB_mse, RGB_mad, OPP_mad, Y_msssim

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
        
        # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
        # GPU and resulting in a slow down.
        with tf.device('/cpu:0'):
            images_lr, images_hr = inputs(files, is_testing=True)
        
        images_sr = model.inference(images_lr, is_training=False)
        ret_loss = list(get_losses(images_hr, images_sr))
        
        # Bicubic upscaling
        shape = tf.shape(images_lr)
        upsize = [shape[-3] * FLAGS.scaling, shape[-2] * FLAGS.scaling]
        images_bicubic = tf.image.resize_images(images_lr, upsize,
                tf.image.ResizeMethod.BICUBIC, align_corners=True)
        
        # restore variables from checkpoint
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.train_dir))
        
        # PNGs output
        ret_pngs = []
        ret_pngs.extend(helper.BatchPNG(images_lr, FLAGS.batch_size))
        ret_pngs.extend(helper.BatchPNG(images_hr, FLAGS.batch_size))
        ret_pngs.extend(helper.BatchPNG(images_sr, FLAGS.batch_size))
        ret_pngs.extend(helper.BatchPNG(images_bicubic, FLAGS.batch_size))
        
        # run session
        sum_loss = [0 for _ in range(len(ret_loss))]
        for i in range(max_steps):
            cur_ret = sess.run(ret_loss + ret_pngs)
            cur_loss = cur_ret[0:len(ret_loss)]
            cur_pngs = cur_ret[len(ret_loss):]
            # monitor losses
            for _ in range(len(ret_loss)):
                sum_loss[_] += cur_loss[_]
            print('batch {}, MSE (RGB) {}, MAD (RGB) {}, MAD (OPP) {}, MS-SSIM (Y) {}'.format(
                   i, cur_loss[0], cur_loss[1], cur_loss[2], cur_loss[3]))
            # images output
            _start = i * FLAGS.batch_size
            _stop = _start + FLAGS.batch_size
            _range = range(_start, _stop)
            ofiles = []
            ofiles.extend([os.path.join(FLAGS.test_dir, '{:0>5}.0.LR.png'.format(_)) for _ in _range])
            ofiles.extend([os.path.join(FLAGS.test_dir, '{:0>5}.1.HR.png'.format(_)) for _ in _range])
            ofiles.extend([os.path.join(FLAGS.test_dir, '{:0>5}.2.SRResNet.png'.format(_)) for _ in _range])
            ofiles.extend([os.path.join(FLAGS.test_dir, '{:0>5}.3.Bicubic.png'.format(_)) for _ in _range])
            helper.WriteFiles(cur_pngs, ofiles)
        sess.close()
        
        # summary
        mean_loss = [l / max_steps for l in sum_loss]
        psnr = 10 * np.log10(1 / mean_loss[0]) if mean_loss[0] > 0 else 100
        print('PSNR (RGB) {}, MAD (RGB) {}, MAD (OPP) {}, MS-SSIM (Y) {}'.format(
               psnr, mean_loss[1], mean_loss[2], mean_loss[3]))

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
