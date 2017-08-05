import sys
import os
import numpy as np
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
                           """Directory where to read checkpoint.""")
tf.app.flags.DEFINE_string('test_dir', './val{}.tmp'.format(FLAGS.postfix),
                           """Directory where to write event logs and test results.""")
tf.app.flags.DEFINE_string('dataset', '../Dataset.MRS/Val',
                           """Directory where stores the dataset.""")
tf.app.flags.DEFINE_integer('var_index', 0,
                            """Index of which the value changes.""")
tf.app.flags.DEFINE_integer('random_seed', 0,
                            """Initialize with specified random seed.""")
tf.app.flags.DEFINE_integer('threads', 8,
                            """Number of threads for Dataset process.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_string('data_format', 'NHWC', # 'NCHW', 'NHWC',
                            """Data layout format.""")
tf.app.flags.DEFINE_integer('seq_size', 2048,
                            """Size of the 1-D sequence.""")
tf.app.flags.DEFINE_integer('num_labels', 16,
                            """Number of labels.""")
tf.app.flags.DEFINE_integer('batch_size', 16,
                            """Batch size.""")
tf.app.flags.DEFINE_float('smoothing', 0.25,
                            """Spatial smoothing for the sequence.""")
tf.app.flags.DEFINE_float('noise_scale', 0.05,
                            """STD of additive Gaussian random noise.""")
tf.app.flags.DEFINE_float('noise_corr', 0.5,
                            """Spatial correlation of the Gaussian random noise.""")
tf.app.flags.DEFINE_float('noise_base', 0.1,
                            """Base ratio of the multiplicative noise.""")
tf.app.flags.DEFINE_float('mse_thresh', 0.001,
                            """MSE lower than this value will be considered as correct prediction.""")
tf.app.flags.DEFINE_float('mad_thresh', 0.02,
                            """MAD lower than this value will be considered as correct prediction.""")

# setup tensorflow
def setup():
    # create session
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options,
        log_device_placement=FLAGS.log_device_placement)
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
def get_losses(labels_gt, labels_pd):
    # losses
    mse = tf.losses.mean_squared_error(labels_gt, labels_pd, weights=1.0)
    mad = tf.losses.absolute_difference(labels_gt, labels_pd, weights=1.0)
    diff = tf.subtract(labels_pd, labels_gt)
    sqr_diff = tf.multiply(diff, diff)
    abs_diff = tf.abs(diff)
    
    # false positive/negative
    mse_batch = tf.reduce_mean(sqr_diff, axis=-1)
    mse_valid = tf.reduce_sum(tf.cast(tf.less(mse_batch, FLAGS.mse_thresh), tf.int32))
    mad_batch = tf.reduce_mean(abs_diff, axis=-1)
    mad_valid = tf.reduce_sum(tf.cast(tf.less(mad_batch, FLAGS.mad_thresh), tf.int32))
    FP = tf.reduce_sum(tf.cast(tf.greater_equal(diff, FLAGS.mad_thresh), tf.int32))
    FN = tf.reduce_sum(tf.cast(tf.less_equal(diff, -FLAGS.mad_thresh), tf.int32))
    
    # error ratio
    epsilon = FLAGS.mad_thresh
    error_ratio = tf.divide(abs_diff, labels_gt + epsilon)
    
    return mse, mad, mse_valid, mad_valid, FP, FN, error_ratio

def total_loss():
    return tf.losses.get_total_loss()

# testing
def test():
    # label names
    if FLAGS.num_labels == 4:
        LABEL_NAMES = ['creatine', 'gaba', 'glutamate', 'glutamine']
    elif FLAGS.num_labels == 12:
        LABEL_NAMES = ['choline-truncated', 'creatine', 'gaba', 'glutamate',
                       'glutamine', 'glycine', 'lactate', 'myo-inositol',
                       'NAAG-truncated', 'n-acetylaspartate', 'phosphocreatine', 'taurine']
    elif FLAGS.num_labels == 16:
        LABEL_NAMES = ['acetate', 'aspartate', 'choline-truncated', 'creatine',
                       'gaba', 'glutamate', 'glutamine', 'histamine',
                       'histidine', 'lactate', 'myo-inositol', 'n-acetylaspartate',
                       'scyllo-inositol', 'succinate', 'taurine', 'valine']
    else:
        LABEL_NAMES = list(range(FLAGS.num_labels))

    # get dataset files
    labels_file = os.path.join(FLAGS.dataset, 'labels\labels.npy')
    files = helper.listdir_files(FLAGS.dataset, recursive=False,
                                 filter_ext=['.npy'],
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
            spectrum, labels_gt = inputs(files, labels_file, epoch_size, is_testing=True)
        
        # model inference and losses
        labels_pd = model.inference(spectrum, is_training=False)
        ret_loss = list(get_losses(labels_gt, labels_pd))
        ret_labels = [labels_gt, labels_pd]
        
        # restore variables from checkpoint
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.train_dir))
        
        # run session
        ret = ret_loss + ret_labels
        ret_loss = ret[:len(ret_loss) - 1]
        sum_loss = [0 for _ in range(len(ret_loss))]
        all_errors = []
        labels_gt = []
        labels_pd = []
        for i in range(max_steps):
            cur_ret = sess.run(ret)
            cur_loss = cur_ret[0:len(ret_loss)]
            cur_errors = cur_ret[len(ret_loss)]
            labels_gt.append(cur_ret[len(ret_loss) + 1])
            labels_pd.append(cur_ret[len(ret_loss) + 2])
            all_errors.append(cur_errors)
            # monitor losses
            for _ in range(len(ret_loss)):
                sum_loss[_] += cur_loss[_]
            #print('batch {}, MSE {}, MAD {}, MSE valid {}, MAD valid {}, False Positives {}, False Negatives {}'.format(i, *cur_loss))
        sess.close()
    
    # summary
    mean_loss = [l / max_steps for l in sum_loss]
    mean_loss[2] /= FLAGS.batch_size
    mean_loss[3] /= FLAGS.batch_size
    mean_loss[4] /= FLAGS.batch_size * FLAGS.num_labels
    mean_loss[5] /= FLAGS.batch_size * FLAGS.num_labels
    print('{} Metabolites'.format(FLAGS.num_labels))
    print('MSE threshold {}'.format(FLAGS.mse_thresh))
    print('MAD threshold {}'.format(FLAGS.mad_thresh))
    print('Totally {} Samples, MSE {}, MAD {}, MSE accuracy {}, MAD accuracy {}, FP rate {}, FN rate {}'.format(epoch_size, *mean_loss))
    
    # errors
    import matplotlib.pyplot as plt
    all_errors = np.concatenate(all_errors, axis=0)
    for _ in range(FLAGS.num_labels):
        errors = all_errors[:, _]
        plt.figure()
        plt.title('Error Ratio Histogram - {}'.format(LABEL_NAMES[_]))
        plt.hist(errors, bins=100, range=(0, 1))
        plt.savefig(os.path.join(FLAGS.test_dir, 'hist{}_{}.png'.format(FLAGS.var_index, _)))
        plt.close()
    
    # labels
    labels_gt = np.concatenate(labels_gt, axis=0)
    labels_pd = np.concatenate(labels_pd, axis=0)
    with open(os.path.join(FLAGS.test_dir, 'labels{}.log'.format(FLAGS.var_index)), mode='w') as file:
        file.write('Labels (Ground Truth)\nLabels (Predicted)\n\n')
        for _ in range(epoch_size):
            file.write('{}\n{}\n\n'.format(labels_gt[_], labels_pd[_]))
    
    # draw plots
    plt.figure()
    plt.title('Predicted Responses to {}'.format(LABEL_NAMES[FLAGS.var_index]))
    x = labels_gt[:, FLAGS.var_index]
    for l in range(FLAGS.num_labels):
        y = labels_pd[:, l]
        plt.plot(x, y, label=LABEL_NAMES[l])
    plt.legend(loc=2)
    plt.savefig(os.path.join(FLAGS.test_dir, 'val{}.png'.format(FLAGS.var_index)))
    plt.close()
    
    print('')

# main
def main(argv=None):
    # arXiv 1509.09308
    # a new class of fast algorithms for convolutional neural networks using Winograd's minimal filtering algorithms
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    
    if not tf.gfile.IsDirectory(FLAGS.train_dir):
        raise FileNotFoundError('Could not find folder {}'.format(FLAGS.train_dir))
    if not tf.gfile.Exists(FLAGS.test_dir):
        tf.gfile.MakeDirs(FLAGS.test_dir)
    test()

if __name__ == '__main__':
    tf.app.run()
