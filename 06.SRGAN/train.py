import sys
import os
from datetime import datetime
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
sys.path.append('..')
from utils import helper

from input import inputs
from model import SRmodel

# working directory
print('Current working directory:\n    {}\n'.format(os.getcwd()))

# flags
FLAGS = tf.app.flags.FLAGS

# parameters
tf.app.flags.DEFINE_string('postfix', '',
                            """Postfix added to train_dir, test_dir, test files, etc.""")
tf.app.flags.DEFINE_string('train_dir', './train{}.tmp'.format(FLAGS.postfix),
                           """Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_boolean('restore', False,
                            """Restore training from checkpoint.""")
tf.app.flags.DEFINE_integer('save_steps', 0,
                            """Number of steps to save meta.""")
tf.app.flags.DEFINE_integer('timeline_steps', 0,
                            """Number of steps to save timeline.""")
tf.app.flags.DEFINE_integer('threads', 8,
                            """Number of threads for Dataset process.""")
tf.app.flags.DEFINE_integer('threads_py', 4,
                            """Number of threads for Dataset process in tf.py_func.""")
tf.app.flags.DEFINE_integer('num_epochs', 20,
                            """Number of epochs to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 100,
                            """Log frequency.""")
tf.app.flags.DEFINE_integer('patch_height', 96,
                            """Block size y.""")
tf.app.flags.DEFINE_integer('patch_width', 96,
                            """Block size x.""")
tf.app.flags.DEFINE_integer('batch_size', 16,
                            """Batch size.""")
tf.app.flags.DEFINE_integer('buffer_size', 8192,
                            """Buffer size for random shuffle.""")
tf.app.flags.DEFINE_boolean('pre_down', False,
                            """Pre-downscale large image for (probably) higher quality data.""")
tf.app.flags.DEFINE_float('color_augmentation', 0.05,
                            """Amount of random color augmentations.""")
tf.app.flags.DEFINE_float('noise_scale', 0.01,
                            """STD of additive Gaussian random noise.""")
tf.app.flags.DEFINE_float('noise_corr', 0.75,
                            """Spatial correlation of the Gaussian random noise.""")
tf.app.flags.DEFINE_boolean('jpeg_coding', True,
                            """Using JPEG to generate compression artifacts for data.""")

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

            format_str = '{}: epoch {}, step {}, g_loss = {:.5}, d_loss = {:.5} ({:.0f} examples/sec; {:.3f} sec/batch)'
            print(format_str.format(datetime.now(), self._epoch, self._step,
                                    loss[0], loss[1], examples_per_sec, sec_per_batch))

# training
def train():
    import random
    files = helper.listdir_files(TRAINSET_PATH,
                                 filter_ext=['.jpeg', '.jpg', '.png'],
                                 encoding=True)
    random.shuffle(files)
    steps_per_epoch = len(files) // FLAGS.batch_size
    epoch_size = steps_per_epoch * FLAGS.batch_size
    max_steps = steps_per_epoch * FLAGS.num_epochs
    files = files[:epoch_size]
    print('epoch size: {}\n{} steps per epoch\n{} epochs\n{} steps'.format(
        epoch_size, steps_per_epoch, FLAGS.num_epochs, max_steps))
    
    with tf.Graph().as_default():
        # pre-processing for input
        with tf.device('/cpu:0'):
            images_lr, images_hr = inputs(FLAGS, files, is_training=True)
        
        # build model
        model = SRmodel(FLAGS, data_format=FLAGS.data_format,
            input_range=FLAGS.input_range, output_range=FLAGS.output_range,
            multiGPU=FLAGS.multiGPU, use_fp16=FLAGS.use_fp16,
            scaling=FLAGS.scaling, image_channels=FLAGS.image_channels,
            input_height=FLAGS.patch_height // FLAGS.scaling,
            input_width=FLAGS.patch_width // FLAGS.scaling)
        
        gd_loss = model.build_train(images_lr, images_hr)
        
        # training step and op
        global_step = tf.contrib.framework.get_or_create_global_step()
        train_op = model.train(global_step)
        
        # a saver object which will save all the variables
        if FLAGS.save_steps > 0:
            saver = tf.train.Saver()
        
        # monitored session
        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(gpu_options=gpu_options,
            log_device_placement=FLAGS.log_device_placement)
        
        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=FLAGS.train_dir,
                hooks=[tf.train.StopAtStepHook(last_step=max_steps),
                       tf.train.NanTensorHook(gd_loss[0]),
                       tf.train.NanTensorHook(gd_loss[1]),
                       LoggerHook(gd_loss, steps_per_epoch)],
                config=config, log_step_count_steps=FLAGS.log_frequency) as mon_sess:
            # options
            if FLAGS.timeline_steps > 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
            # run session
            while not mon_sess.should_stop():
                mon_sess.run(train_op)
                # not work on MonitoredSession
                '''
                step = tf.train.global_step(sess, global_step)
                if FLAGS.timeline_steps > 0 and step % FLAGS.timeline_steps == 0:
                    sess.run(train_op, options=run_options, run_metadata=run_metadata)
                    # Create the Timeline object, and write it to a json
                    tl = timeline.Timeline(run_metadata.step_stats)
                    ctf = tl.generate_chrome_trace_format()
                    with open(os.path.join(FLAGS.train_dir, 'timeline_{:0>7}.json'.format(step)), 'a') as f:
                        f.write(ctf)
                else:
                    sess.run(train_op)
                if FLAGS.save_steps > 0 and step % FLAGS.save_steps == 0:
                    saver.save(sess, os.path.join(FLAGS.train_dir, 'model_{:0>7}'.format(step)),
                               write_meta_graph=True, write_state=True)
                '''

# main
def main(argv=None):
    # arXiv 1509.09308
    # a new class of fast algorithms for convolutional neural networks using Winograd's minimal filtering algorithms
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    
    if not FLAGS.restore:
        if tf.gfile.Exists(FLAGS.train_dir):
            tf.gfile.DeleteRecursively(FLAGS.train_dir)
        tf.gfile.MakeDirs(FLAGS.train_dir)
    train()

if __name__ == '__main__':
    tf.app.run()
