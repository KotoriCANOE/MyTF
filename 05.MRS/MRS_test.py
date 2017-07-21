import sys
import os
import tensorflow as tf
sys.path.append('..')
from utils import helper

from MRS_input import inputs
import MRS as model

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
tf.app.flags.DEFINE_integer('seq_size', 2048,
                            """Size of the 1-D sequence.""")
tf.app.flags.DEFINE_integer('batch_size', 16,
                            """Batch size.""")

# constants
TESTSET_PATH = r'..\Dataset.MRS\Test'

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
def get_losses(labels_gt, labels_pd):
    mse = tf.losses.mean_squared_error(labels_gt, labels_pd, weights=1.0)
    mad = tf.losses.absolute_difference(labels_gt, labels_pd, weights=1.0)
    return mse, mad

def total_loss():
    return tf.losses.get_total_loss()

# testing
def test():
    labels_file = os.path.join(TRAINSET_PATH, 'labels.npy0')
    files = helper.listdir_files(TRAINSET_PATH,
                                 filter_ext=['.npy'],
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
            spectrum, labels_gt = inputs(files, labels_file, is_training=False)
        
        labels_pd = model.inference(spectrum, is_training=False)
        ret_loss = [get_losses(labels_gt, labels_pd)]
        
        # restore variables from checkpoint
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.train_dir))
        
        # run session
        sum_loss = [0 for _ in range(len(ret_loss))]
        for i in range(max_steps):
            cur_ret = sess.run(ret_loss)
            cur_loss = cur_ret[0:len(ret_loss)]
            # monitor losses
            for _ in range(len(ret_loss)):
                sum_loss[_] += cur_loss[_]
            print('batch {}, MSE {}, MAD {}'.format(
                   i, cur_loss[0], cur_loss[1]))
        sess.close()
        
        # summary
        mean_loss = [l / max_steps for l in sum_loss]
        print('MSE {}, MAD {}'.format(
               mean_loss[0], mean_loss[1]))

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
