import sys
import os
import numpy as np
import tensorflow as tf
sys.path.append('..')
from utils import helper

from input import inputs
from model import MRSmodel

# working directory
print('Current working directory:\n    {}\n'.format(os.getcwd()))

# flags
FLAGS = tf.app.flags.FLAGS

# parameters
tf.app.flags.DEFINE_string('postfix', '',
                            """Postfix added to train_dir, test_dir, test files, etc.""")
tf.app.flags.DEFINE_string('train_dir', './train{}.tmp'.format(FLAGS.postfix),
                           """Directory where to read checkpoint.""")
tf.app.flags.DEFINE_string('test_dir', './test{}.tmp'.format(FLAGS.postfix),
                           """Directory where to write event logs and test results.""")
tf.app.flags.DEFINE_string('dataset', '../../Dataset.MRS/Test2',
                           """Directory where stores the dataset.""")
tf.app.flags.DEFINE_boolean('progress', False,
                            """Whether to test across the entire training procedure.""")
tf.app.flags.DEFINE_integer('random_seed', 0,
                            """Initialize with specified random seed.""")
tf.app.flags.DEFINE_integer('threads', 8,
                            """Number of threads for Dataset process.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('batch_size', 16,
                            """Batch size.""")
tf.app.flags.DEFINE_float('smoothing', 0.5,
                            """Spatial smoothing for the sequence.""")
tf.app.flags.DEFINE_float('noise_scale', 0.03,
                            """STD of additive Gaussian random noise.""")
tf.app.flags.DEFINE_float('noise_corr', 0.5,
                            """Spatial correlation of the Gaussian random noise.""")
tf.app.flags.DEFINE_float('noise_base', 0.1,
                            """Base ratio of the multiplicative noise.""")
tf.app.flags.DEFINE_float('mse_thresh', 0.005,
                            """MSE lower than this value will be considered as correct prediction.""")
tf.app.flags.DEFINE_float('mad_thresh', 0.05,
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
def get_losses(ref, pred):
    # losses
    mse = tf.losses.mean_squared_error(ref, pred, weights=1.0)
    mad = tf.losses.absolute_difference(ref, pred, weights=1.0)
    diff = tf.subtract(pred, ref)
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
    error_ratio = tf.divide(abs_diff, ref + epsilon)
    
    #return each loss
    return mse, mad, mse_valid, mad_valid, FP, FN, error_ratio

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
    
    print(*LABEL_NAMES)
    print('No.{}'.format(FLAGS.postfix))

    # get dataset files
    labels_file = os.path.join(FLAGS.dataset, 'labels/labels.npy')
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
            spectrum, labels_ref = inputs(FLAGS, files, labels_file, epoch_size, is_testing=True)
        
        # build model
        model = MRSmodel(FLAGS, data_format=FLAGS.data_format,
            seq_size=FLAGS.seq_size, num_labels=FLAGS.num_labels)
        
        model.build_model(spectrum)
        
        # get output
        labels_pred = tf.get_default_graph().get_tensor_by_name('Output:0')
        
        # losses
        ret_loss = list(get_losses(labels_ref, labels_pred))
        ret_labels = [labels_ref, labels_pred]
        ret = ret_loss + ret_labels
        ret_loss = ret[:len(ret_loss) - 1]
        
        # model files
        if FLAGS.progress:
            mfiles = helper.listdir_files(FLAGS.train_dir, recursive=False,
                                          filter_ext=['.index'],
                                          encoding=None)
            mfiles = [f[:-6] for f in mfiles if 'model_' in f]
            mfiles.sort()
            stats = []
        else:
            mfiles = [tf.train.latest_checkpoint(FLAGS.train_dir)]
        
        for model_file in mfiles:
            # restore variables from checkpoint
            saver = tf.train.Saver()
            saver.restore(sess, model_file)
            
            # run session
            sum_loss = [0 for _ in range(len(ret_loss))]
            all_errors = []
            labels_ref = []
            labels_pred = []
            for i in range(max_steps):
                cur_ret = sess.run(ret)
                cur_loss = cur_ret[0:len(ret_loss)]
                cur_errors = cur_ret[len(ret_loss)]
                labels_ref.append(cur_ret[len(ret_loss) + 1])
                labels_pred.append(cur_ret[len(ret_loss) + 2])
                all_errors.append(cur_errors)
                # monitor losses
                for _ in range(len(ret_loss)):
                    sum_loss[_] += cur_loss[_]
                #print('batch {}, MSE {}, MAD {}, MSE valid {}, MAD valid {}, False Positives {}, False Negatives {}'.format(i, *cur_loss))
            
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
            
            # save stats
            if FLAGS.progress:
                model_num = os.path.split(model_file)[1][6:]
                stats.append(np.array([float(model_num)] + mean_loss))
        
        sess.close()
    
    # errors
    import matplotlib.pyplot as plt
    all_errors = np.concatenate(all_errors, axis=0)
    for _ in range(FLAGS.num_labels):
        errors = all_errors[:, _]
        plt.figure()
        plt.title('Error Ratio Histogram - {}'.format(LABEL_NAMES[_]))
        plt.hist(errors, bins=100, range=(0, 1))
        plt.savefig(os.path.join(FLAGS.test_dir, 'hist_{}.png'.format(_)))
        plt.close()
    
    # labels
    labels_ref = np.concatenate(labels_ref, axis=0)
    labels_pred = np.concatenate(labels_pred, axis=0)
    with open(os.path.join(FLAGS.test_dir, 'labels.log'), mode='w') as file:
        file.write('Labels (Ground Truth)\nLabels (Predicted)\n\n')
        for _ in range(epoch_size):
            file.write('{}\n{}\n\n'.format(labels_ref[_], labels_pred[_]))
    
    # save stats
    if FLAGS.progress:
        stats = np.stack(stats)
        np.save(os.path.join(FLAGS.test_dir, 'stats.npy'), stats)
        fig, ax = plt.subplots()
        ax.set_title('Test Error with Training Progress')
        ax.set_xlabel('training steps')
        ax.set_ylabel('mean absolute difference')
        ax.set_xscale('linear')
        ax.set_yscale('log')
        ax.plot(stats[:, 0], stats[:, 2])
        ax.axis(ymin=0)
        plt.tight_layout()
        plt.savefig(os.path.join(FLAGS.test_dir, 'stats.png'))
        plt.close()
    
    print('')

# main
def main(argv=None):
    # arXiv 1509.09308
    # a new class of fast algorithms for convolutional neural networks using Winograd's minimal filtering algorithms
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    
    if not tf.gfile.IsDirectory(FLAGS.train_dir):
        raise FileNotFoundError('Could not find folder {}'.format(FLAGS.train_dir))
    if tf.gfile.Exists(FLAGS.test_dir):
        tf.gfile.DeleteRecursively(FLAGS.test_dir)
    tf.gfile.MakeDirs(FLAGS.test_dir)
    test()

if __name__ == '__main__':
    tf.app.run()
