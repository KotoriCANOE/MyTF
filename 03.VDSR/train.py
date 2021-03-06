import sys
import os
from datetime import datetime
import time
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
tf.app.flags.DEFINE_string('pretrain_dir', '',
                           """Directory where to load pre-trained model.""")
tf.app.flags.DEFINE_string('dataset', '../../Dataset.SR/Train',
                           """Directory where stores the dataset.""")
tf.app.flags.DEFINE_boolean('restore', False,
                            """Restore training from checkpoint.""")
tf.app.flags.DEFINE_integer('save_steps', 5000,
                            """Number of steps to save meta.""")
tf.app.flags.DEFINE_integer('timeline_steps', 911,
                            """Number of steps to save timeline.""")
tf.app.flags.DEFINE_integer('threads', 8,
                            """Number of threads for Dataset process.""")
tf.app.flags.DEFINE_integer('threads_py', 8,
                            """Number of threads for Dataset process in tf.py_func.""")
tf.app.flags.DEFINE_integer('num_epochs', 20,
                            """Number of epochs to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 500,
                            """Log frequency.""")
tf.app.flags.DEFINE_integer('patch_height', 96,
                            """Block size y.""")
tf.app.flags.DEFINE_integer('patch_width', 96,
                            """Block size x.""")
tf.app.flags.DEFINE_integer('batch_size', 16,
                            """Batch size.""")
tf.app.flags.DEFINE_integer('buffer_size', 65536,
                            """Buffer size for random shuffle.""")
tf.app.flags.DEFINE_boolean('pre_down', False,
                            """Pre-downscale large image for (probably) higher quality data.""")
tf.app.flags.DEFINE_float('color_augmentation', 0.05,
                            """Amount of random color augmentations.""")
tf.app.flags.DEFINE_float('random_resizer', 0,
                            """value for resizer choice, 0 for random resizer.""")
tf.app.flags.DEFINE_float('noise_scale', 0.01,
                            """STD of additive Gaussian random noise.""")
tf.app.flags.DEFINE_float('noise_corr', 0.75,
                            """Spatial correlation of the Gaussian random noise.""")
tf.app.flags.DEFINE_boolean('jpeg_coding', True,
                            """Using JPEG to generate compression artifacts for data.""")

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

            format_str = '{}: epoch {}, step {}, g_loss = {:.5} ({:.0f} samples/sec; {:.3f} sec/batch)'
            print(format_str.format(datetime.now(), self._epoch, self._step,
                                    loss, examples_per_sec, sec_per_batch))

# training
def train():
    import random
    files = helper.listdir_files(FLAGS.dataset,
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
            scaling=FLAGS.scaling, image_channels=FLAGS.image_channels)
        
        g_loss = model.build_train(images_lr, images_hr)
        
        # training step and op
        global_step = tf.train.get_or_create_global_step()
        g_train_op = model.train(global_step)
        
        # a saver object which will save all the variables
        saver = tf.train.Saver(var_list=model.g_svars,
            max_to_keep=1 << 16, save_relative_paths=True)
        
        # save the graph
        saver.export_meta_graph(os.path.join(FLAGS.train_dir, 'model.meta'),
            as_text=False, clear_devices=True, clear_extraneous_savers=True)
        
        # monitored session
        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(gpu_options=gpu_options,
            log_device_placement=FLAGS.log_device_placement)
        
        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=FLAGS.train_dir,
                hooks=[tf.train.StopAtStepHook(last_step=max_steps),
                       tf.train.NanTensorHook(g_loss),
                       LoggerHook(g_loss, steps_per_epoch)],
                config=config, log_step_count_steps=FLAGS.log_frequency) as mon_sess:
            # options
            sess = helper.get_session(mon_sess)
            if FLAGS.timeline_steps > 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
            # restore pre-trained model
            if FLAGS.pretrain_dir and not FLAGS.restore:
                saver.restore(sess, os.path.join(FLAGS.pretrain_dir, 'model'))
            # sessions
            def run_sess(options=None, run_metadata=None):
                mon_sess.run(g_train_op, options=options, run_metadata=run_metadata)
            # run session
            while not mon_sess.should_stop():
                step = tf.train.global_step(sess, global_step)
                if FLAGS.timeline_steps > 0 and step // FLAGS.timeline_steps < 10 and step % FLAGS.timeline_steps == 0:
                    run_sess(run_options, run_metadata)
                    # Create the Timeline object, and write it to a json
                    tl = timeline.Timeline(run_metadata.step_stats)
                    ctf = tl.generate_chrome_trace_format()
                    with open(os.path.join(FLAGS.train_dir, 'timeline_{:0>7}.json'.format(step)), 'a') as f:
                        f.write(ctf)
                else:
                    run_sess()
                if FLAGS.save_steps > 0 and step % FLAGS.save_steps == 0:
                    saver.save(sess, os.path.join(FLAGS.train_dir, 'model_{:0>7}'.format(step)),
                               write_meta_graph=False, write_state=False)

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
