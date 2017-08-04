import sys
import os
from datetime import datetime
import time
import tensorflow as tf
sys.path.append('..')
from utils import helper

from MRS_input import inputs
import MRS as model

# working directory
print('Current working directory:\n    {}\n'.format(os.getcwd()))

# flags
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('postfix', '',
                            """Postfix added to train_dir, test_dir, test files, etc.""")
tf.app.flags.DEFINE_string('train_dir', './train{}.tmp'.format(FLAGS.postfix),
                           """Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_string('dataset', '../Dataset.MRS/Train',
                           """Directory where stores the dataset.""")
tf.app.flags.DEFINE_boolean('restore', False,
                            """Restore training from checkpoint.""")
tf.app.flags.DEFINE_integer('threads', 8,
                            """Number of threads for Dataset process.""")
tf.app.flags.DEFINE_integer('epoch_size', 0,
                            """Number of samples in an epoch.""")
tf.app.flags.DEFINE_integer('num_epochs', 40,
                            """Number of epochs to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 1000,
                            """Log frequency.""")
tf.app.flags.DEFINE_string('data_format', 'NHWC', # 'NCHW'
                            """Data layout format.""")
tf.app.flags.DEFINE_integer('seq_size', 2048,
                            """Size of the 1-D sequence.""")
tf.app.flags.DEFINE_integer('num_labels', 16,
                            """Number of labels.""")
tf.app.flags.DEFINE_integer('batch_size', 16,
                            """Batch size.""")
tf.app.flags.DEFINE_integer('buffer_size', 8192,
                            """Buffer size for random shuffle.""")
tf.app.flags.DEFINE_float('smoothing', 0.25,
                            """Spatial smoothing for the sequence.""")
tf.app.flags.DEFINE_float('noise_scale', 0.05,
                            """STD of additive Gaussian random noise.""")
tf.app.flags.DEFINE_float('noise_corr', 0.5,
                            """Spatial correlation of the Gaussian random noise.""")
tf.app.flags.DEFINE_float('noise_base', 0.1,
                            """Base ratio of the multiplicative noise.""")

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

            mad = run_values.results
            examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
            sec_per_batch = float(duration / FLAGS.log_frequency)

            format_str = '{}: epoch {}, step {}, MAD = {:.5} ({:.0f} examples/sec; {:.3f} sec/batch)'
            print(format_str.format(datetime.now(), self._epoch, self._step,
                                    mad, examples_per_sec, sec_per_batch))

# losses
def loss_mse(labels_gt, labels_pd):
    mse = tf.losses.mean_squared_error(labels_gt, labels_pd, weights=1.0)
    return mse

def loss_mad(labels_gt, labels_pd):
    mad = tf.losses.absolute_difference(labels_gt, labels_pd, weights=1.0)
    return mad

def total_loss():
    return tf.losses.get_total_loss()

# training
def train():
    labels_file = os.path.join(FLAGS.dataset, 'labels\labels.npy')
    files = helper.listdir_files(FLAGS.dataset, recursive=False,
                                 filter_ext=['.npy'],
                                 encoding='utf-8')
    epoch_size = FLAGS.epoch_size if FLAGS.epoch_size > 0 else len(files)
    steps_per_epoch = epoch_size // FLAGS.batch_size
    epoch_size = steps_per_epoch * FLAGS.batch_size
    max_steps = steps_per_epoch * FLAGS.num_epochs
    files = files[:epoch_size]
    print('epoch size: {}\n{} steps per epoch\n{} epochs\n{} steps'.format(
        epoch_size, steps_per_epoch, FLAGS.num_epochs, max_steps))
    
    with tf.Graph().as_default():
        # pre-processing for input
        with tf.device('/cpu:0'):
            spectrum, labels_gt = inputs(files, labels_file, epoch_size, is_training=True)
        
        # model inference and losses
        labels_pd = model.inference(spectrum, is_training=True)
        main_loss = loss_mad(labels_gt, labels_pd)
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
            # run session
            while not mon_sess.should_stop():
                mon_sess.run(train_op)

# main
def main(argv=None):
    if not FLAGS.restore:
        if tf.gfile.Exists(FLAGS.train_dir):
            tf.gfile.DeleteRecursively(FLAGS.train_dir)
        tf.gfile.MakeDirs(FLAGS.train_dir)
    train()

if __name__ == '__main__':
    tf.app.run()
