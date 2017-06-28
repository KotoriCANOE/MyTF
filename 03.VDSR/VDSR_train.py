import sys
import os
from datetime import datetime
import time
import math
import tensorflow as tf

from VDSR_input import inputs
import VDSR

# working directory
print('Current working directory:\n    {}\n'.format(os.getcwd()))
THIS_FILE_NAME = os.path.splitext(os.path.basename(sys.argv[0]))[0]

# flags
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', './{}.tmp'.format(THIS_FILE_NAME),
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 100000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 100,
                            """Log frequency.""")
tf.app.flags.DEFINE_integer('batch_size', 64,
                            """Batch size.""")

# constants
TRAIN_FILE = r'D:\Datasets\91-image\linearscale2_bicubic_point\train_42_6.h5'
TEST_FILE = r'D:\Datasets\91-image\linearscale2_bicubic_point\test_160_124.h5'

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
    with tf.Graph().as_default():
        # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
        # GPU and resulting in a slow down.
        with tf.device('/cpu:0'):
            images_lr, images_up, images_hr, epoch_size = inputs(TRAIN_FILE, FLAGS.batch_size, shuffle=True)
        steps_per_epoch = epoch_size / FLAGS.batch_size
        print('{} steps per epoch'.format(steps_per_epoch))
        
        images_sr = VDSR.inference(images_lr, images_up)
        main_loss = VDSR.main_loss(images_sr, images_hr)
        loss = VDSR.loss()
        
        global_step = tf.contrib.framework.get_or_create_global_step()
        train_op = VDSR.train(loss, global_step, epoch_size)
        
        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=FLAGS.train_dir,
                hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                       tf.train.NanTensorHook(loss),
                       LoggerHook(main_loss, steps_per_epoch)],
                config=tf.ConfigProto(
                        log_device_placement=FLAGS.log_device_placement)) as mon_sess:
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
