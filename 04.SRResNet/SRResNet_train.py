import sys
import os
from datetime import datetime
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
sys.path.append('..')
from utils import helper
import utils.image

from SRResNet_input import inputs
import SRResNet as model

# working directory
print('Current working directory:\n    {}\n'.format(os.getcwd()))

# flags
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('postfix', '',
                            """Postfix added to train_dir, test_dir, test files, etc.""")
tf.app.flags.DEFINE_string('train_dir', './train{}.tmp'.format(FLAGS.postfix),
                           """Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_boolean('restore', False,
                            """Restore training from checkpoint.""")
tf.app.flags.DEFINE_integer('threads', 8,
                            """Number of threads for Dataset process.""")
tf.app.flags.DEFINE_integer('num_epochs', 20,
                            """Number of epochs to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 100,
                            """Log frequency.""")
tf.app.flags.DEFINE_string('data_format', 'NCHW', # 'NHWC'
                            """Data layout format.""")
tf.app.flags.DEFINE_integer('patch_height', 96,
                            """Block size y.""")
tf.app.flags.DEFINE_integer('patch_width', 96,
                            """Block size x.""")
tf.app.flags.DEFINE_integer('batch_size', 16,
                            """Batch size.""")
tf.app.flags.DEFINE_integer('buffer_size', 8192,
                            """Buffer size for random shuffle.""")
tf.app.flags.DEFINE_float('color_augmentation', 0.05,
                            """Amount of random color augmentations.""")
tf.app.flags.DEFINE_float('noise_level', 0,
                            """STD of additive normal dist. random noise.""")
tf.app.flags.DEFINE_float('mixed_alpha', 0.50,
                            """blend weight for mixed loss.""")

# constants
TRAINSET_PATH = r'..\Dataset.SR\Train'

# helper class
class LoggerHook(tf.train.SessionRunHook):
    """Logs loss and runtime."""
    def __init__(self, fetches, steps_per_epoch):
        self.fetches = fetches
        self.steps_per_epoch = steps_per_epoch

    def begin(self):
        self._epoch = -1
        self._step = -1
        self._start_time = time.time()

    def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(self.fetches)  # asks for fetches

    def after_run(self, run_context, run_values):
        if self._step % self.steps_per_epoch < 1:
            self._epoch += 1
        if self._step % FLAGS.log_frequency == 0:
            current_time = time.time()
            duration = current_time - self._start_time
            self._start_time = current_time

            loss = run_values.results
            examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
            sec_per_batch = float(duration / FLAGS.log_frequency)

            format_str = '{}: epoch {}, step {}, loss = {:.5} ({:.0f} examples/sec; {:.3f} sec/batch)'
            print(format_str.format(datetime.now(), self._epoch, self._step,
                                    loss, examples_per_sec, sec_per_batch))

# losses
def loss_mse(gtruth, pred, weights=1.0):
    # RGB color space
    mse = tf.losses.mean_squared_error(gtruth, pred, weights=weights)
    return mse

def loss_mad(gtruth, pred, weights=1.0):
    # RGB color space
    RGB_mad = tf.losses.absolute_difference(gtruth, pred, weights=weights)
    return RGB_mad

def loss_mixed1(gtruth, pred, alpha=0.50, weights1=1.0, weights2=1.0):
    weights1 *= 1 - alpha
    weights2 *= alpha
    RGB_mad = tf.losses.absolute_difference(gtruth, pred, weights=weights1)
    # OPP color space - Y
    Y_gtruth = utils.image.RGB2Y(gtruth, data_format=FLAGS.data_format)
    Y_pred = utils.image.RGB2Y(pred, data_format=FLAGS.data_format)
    Y_ss_ssim = (1 - utils.image.SS_SSIM(Y_gtruth, Y_pred, data_format=FLAGS.data_format)) * weights2
    tf.losses.add_loss(Y_ss_ssim)
    return RGB_mad + Y_ss_ssim

def loss_mixed2(gtruth, pred, alpha=0.50, weights1=1.0, weights2=1.0):
    weights1 *= 1 - alpha
    weights2 *= alpha
    RGB_mad = tf.losses.absolute_difference(gtruth, pred, weights=weights1)
    # OPP color space - Y
    Y_gtruth = utils.image.RGB2Y(gtruth, data_format=FLAGS.data_format)
    Y_pred = utils.image.RGB2Y(pred, data_format=FLAGS.data_format)
    Y_ms_ssim = (1 - utils.image.MS_SSIM2(Y_gtruth, Y_pred, sigma=[0.6,1.5,4],
                norm=False, data_format=FLAGS.data_format)) * weights2
    tf.losses.add_loss(Y_ms_ssim)
    return RGB_mad + Y_ms_ssim

def total_loss():
    return tf.losses.get_total_loss()

# training
def train():
    files = helper.listdir_files(TRAINSET_PATH,
                                 filter_ext=['.jpeg', '.jpg', '.png'],
                                 encoding=True)
    steps_per_epoch = len(files) // FLAGS.batch_size
    epoch_size = steps_per_epoch * FLAGS.batch_size
    max_steps = steps_per_epoch * FLAGS.num_epochs
    files = files[:epoch_size]
    print('epoch size: {}\n{} steps per epoch\n{} epochs\n{} steps'.format(
        epoch_size, steps_per_epoch, FLAGS.num_epochs, max_steps))
    
    with tf.Graph().as_default():
        # pre-processing for input
        with tf.device('/cpu:0'):
            images_lr, images_hr = inputs(files, is_training=True)
        
        # model inference and losses
        images_sr = model.inference(images_lr, is_training=True)
        main_loss = loss_mixed2(images_hr, images_sr, alpha=FLAGS.mixed_alpha)
        train_loss = total_loss()
        
        # training step and op
        global_step = tf.contrib.framework.get_or_create_global_step()
        train_op = model.train(train_loss, global_step)
        
        # monitored session
        config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
        config.gpu_options.allow_growth = True
        
        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=FLAGS.train_dir,
                hooks=[tf.train.StopAtStepHook(last_step=max_steps),
                       tf.train.NanTensorHook(train_loss),
                       LoggerHook(main_loss, steps_per_epoch)],
                config=config, log_step_count_steps=FLAGS.log_frequency) as mon_sess:
            # options
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            step = 0
            # run session
            while not mon_sess.should_stop():
                if step % 5000 == 0:
                    mon_sess.run(train_op, options=run_options, run_metadata=run_metadata)
                    # Create the Timeline object, and write it to a json
                    tl = timeline.Timeline(run_metadata.step_stats)
                    ctf = tl.generate_chrome_trace_format()
                    with open(os.path.join(FLAGS.train_dir, 'timeline_{:0>7}.json'.format(step)), 'a') as f:
                        f.write(ctf)
                else:
                    mon_sess.run(train_op)
                step += 1

# main
def main(argv=None):
    if not FLAGS.restore:
        if tf.gfile.Exists(FLAGS.train_dir):
            tf.gfile.DeleteRecursively(FLAGS.train_dir)
        tf.gfile.MakeDirs(FLAGS.train_dir)
    train()

if __name__ == '__main__':
    tf.app.run()
