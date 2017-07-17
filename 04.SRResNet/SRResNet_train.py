import sys
import os
from datetime import datetime
import time
import tensorflow as tf
sys.path.append('..')
from utils import helper

from SRResNet_input import inputs
import SRResNet

# working directory
print('Current working directory:\n    {}\n'.format(os.getcwd()))
THIS_FILE_NAME = os.path.splitext(os.path.basename(sys.argv[0]))[0]

# flags
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', './{}.tmp'.format(THIS_FILE_NAME),
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('threads', 4,
                            """Number of threads for Dataset process.""")
tf.app.flags.DEFINE_integer('num_epochs', 20,
                            """Number of epochs to run.""")
#tf.app.flags.DEFINE_integer('max_steps', 1e5,
#                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 100,
                            """Log frequency.""")
tf.app.flags.DEFINE_integer('block_size', 128,
                            """Block size.""")
tf.app.flags.DEFINE_integer('batch_size', 16,
                            """Batch size.""")
tf.app.flags.DEFINE_integer('buffer_size', 8192,
                            """Buffer size for random shuffle.""")

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

            #import math
            #mse = run_values.results
            #psnr = 10 * math.log10(1 / mse) if mse > 0 else 100
            mad = run_values.results
            examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
            sec_per_batch = float(duration / FLAGS.log_frequency)

            #format_str = '{}: epoch {}, step {}, PSNR = {:.3f} ({:.0f} examples/sec; {:.3f} sec/batch)'
            format_str = '{}: epoch {}, step {}, MAD = {:.5} ({:.0f} examples/sec; {:.3f} sec/batch)'
            print(format_str.format(datetime.now(), self._epoch, self._step,
                                    mad, examples_per_sec, sec_per_batch))

# training
def train():
    files = helper.listdir_files(TRAINSET_PATH,
                                 filter_ext=['.jpeg', '.jpg', '.png'],
                                 encoding=True)
    steps_per_epoch = len(files) // FLAGS.batch_size
    epoch_size = steps_per_epoch * FLAGS.batch_size
    files = files[:epoch_size]
    max_steps = int(steps_per_epoch * FLAGS.num_epochs)
    print('epoch size: {}\n{} steps per epoch\n{} epochs\n{} steps'.format(
        epoch_size, steps_per_epoch, FLAGS.num_epochs, max_steps))
    
    with tf.Graph().as_default():
        # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
        # GPU and resulting in a slow down.
        with tf.device('/cpu:0'):
            images_lr, images_hr = inputs(files)
        
        images_sr = SRResNet.inference(images_lr, is_training=True)
        main_loss = SRResNet.main_loss(images_hr, images_sr)
        loss = SRResNet.loss()
        
        global_step = tf.contrib.framework.get_or_create_global_step()
        train_op = SRResNet.train(loss, global_step)
        
        config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
        config.gpu_options.allow_growth = True
        
        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=FLAGS.train_dir,
                hooks=[tf.train.StopAtStepHook(last_step=max_steps),
                       tf.train.NanTensorHook(loss),
                       LoggerHook(main_loss[0], steps_per_epoch)],
                config=config) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)

# main
def main(argv=None):
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()

if __name__ == '__main__':
    tf.app.run()
