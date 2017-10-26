import sys
import os
import numpy as np
import tensorflow as tf
sys.path.append('..')
from utils import helper
import utils.image

from input import inputs
from model import ICmodel

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
tf.app.flags.DEFINE_string('dataset', '../../Dataset.SR/Test',
                           """Directory where stores the dataset.""")
tf.app.flags.DEFINE_boolean('progress', False,
                            """Whether to test across the entire training procedure.""")
tf.app.flags.DEFINE_integer('random_seed', 0,
                            """Initialize with specified random seed.""")
tf.app.flags.DEFINE_integer('threads', 8,
                            """Number of threads for Dataset process.""")
tf.app.flags.DEFINE_integer('threads_py', 8,
                            """Number of threads for Dataset process in tf.py_func.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('patch_height', 512,
                            """Block size y.""")
tf.app.flags.DEFINE_integer('patch_width', 512,
                            """Block size x.""")
tf.app.flags.DEFINE_integer('batch_size', 1,
                            """Batch size.""")
tf.app.flags.DEFINE_boolean('pre_down', True,
                            """Pre-downscale large image for (probably) higher quality data.""")
tf.app.flags.DEFINE_integer('multistage_resize', 0,
                            """Apply resizer (n * 2 + 1) times to simulate more complex filtered data.""")
tf.app.flags.DEFINE_float('random_resizer', 0,
                            """value for resizer choice, 0 for random resizer.""")
tf.app.flags.DEFINE_float('noise_scale', 0.01,
                            """STD of additive Gaussian random noise.""")
tf.app.flags.DEFINE_float('noise_corr', 0.75,
                            """Spatial correlation of the Gaussian random noise.""")
tf.app.flags.DEFINE_float('jpeg_coding', 0.0,
                            """Using JPEG to generate compression artifacts for data.""")

# reset random seeds
def reset_random(seed=0):
    import random
    tf.set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

# setup tensorflow and return session
def setup():
    # create session
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=config)

    # initialize rng with a deterministic seed
    reset_random(FLAGS.random_seed)

    #summary_writer = tf.summary.FileWriter(FLAGS.test_dir, sess.graph)
    #return sess, summary_writer
    return sess

# losses
def get_losses(ref, pred, enc):
    # quantization loss
    sparsity = tf.count_nonzero(enc, dtype=tf.float32) / tf.cast(tf.reduce_prod(tf.shape(enc)), tf.float32)

    # RGB color space
    RGB_mse = tf.losses.mean_squared_error(ref, pred, weights=1.0)
    RGB_mad = tf.losses.absolute_difference(ref, pred, weights=1.0)
    
    # OPP color space - Y
    Y_ref = utils.image.RGB2Y(ref, data_format=FLAGS.data_format)
    Y_pred = utils.image.RGB2Y(pred, data_format=FLAGS.data_format)
    Y_ss_ssim = utils.image.SS_SSIM(Y_ref, Y_pred, data_format=FLAGS.data_format)
    Y_ms_ssim = utils.image.MS_SSIM2(Y_ref, Y_pred, norm=True, data_format=FLAGS.data_format)
    
    # return each loss
    return RGB_mse, RGB_mad, Y_ss_ssim, Y_ms_ssim, sparsity

# testing
def test():
    files = helper.listdir_files(FLAGS.dataset,
                                 filter_ext=['.jpeg', '.jpg', '.png'],
                                 encoding=True)
    steps_per_epoch = len(files) // FLAGS.batch_size
    epoch_size = steps_per_epoch * FLAGS.batch_size
    max_steps = steps_per_epoch
    files = files[:epoch_size]
    
    with tf.Graph().as_default():
        # pre-processing for input
        with tf.device('/cpu:0'):
            images_src = inputs(FLAGS, files, is_testing=True)
        
        # build model
        model = ICmodel(FLAGS, data_format=FLAGS.data_format,
            input_range=FLAGS.input_range, output_range=FLAGS.output_range,
            qp_range=FLAGS.qp_range, multiGPU=FLAGS.multiGPU, use_fp16=FLAGS.use_fp16,
            image_channels=FLAGS.image_channels)
        
        model.build_model(images_src)
        
        # a saver object which will save all the variables
        saver = tf.train.Saver(var_list=model.g_svars)
        
        # get output
        images_enc = tf.get_default_graph().get_tensor_by_name('Encoded:0')
        images_dec = tf.get_default_graph().get_tensor_by_name('Decoded:0')
        
        # losses
        ret_loss = list(get_losses(images_src, images_dec, images_enc))
        
        # post-processing for output
        with tf.device('/cpu:0'):
            # data format conversion
            if FLAGS.data_format == 'NCHW':
                images_src = utils.image.NCHW2NHWC(images_src)
                images_dec = utils.image.NCHW2NHWC(images_dec)
                images_enc = utils.image.NCHW2NHWC(images_enc)
            
            # PNGs output
            ret_pngs = []
            ret_pngs.extend(helper.BatchPNG(images_src, FLAGS.batch_size))
            ret_pngs.extend(helper.BatchPNG(images_dec, FLAGS.batch_size))
            ret_pngs.extend(helper.BatchPNG(images_enc, FLAGS.batch_size))
        
        # test latest checkpoint
        with setup() as sess:
            # restore variables from latest checkpoint
            model_file = tf.train.latest_checkpoint(FLAGS.train_dir)
            saver.restore(sess, model_file)
            
            # run session
            ret = ret_loss + ret_pngs
            sum_loss = [0 for _ in range(len(ret_loss))]
            for step in range(max_steps):
                cur_ret = sess.run(ret)
                cur_loss = cur_ret[0:len(ret_loss)]
                cur_pngs = cur_ret[len(ret_loss):]
                # monitor losses
                for _ in range(len(ret_loss)):
                    sum_loss[_] += cur_loss[_]
                # images output
                _start = step * FLAGS.batch_size
                _stop = _start + FLAGS.batch_size
                _range = range(_start, _stop)
                ofiles = []
                ofiles.extend([os.path.join(FLAGS.test_dir,
                    '{:0>5}.0.src.png'.format(_)) for _ in _range])
                ofiles.extend([os.path.join(FLAGS.test_dir,
                    '{:0>5}.1.dec{}.png'.format(_, FLAGS.postfix)) for _ in _range])
                ofiles.extend([os.path.join(FLAGS.test_dir,
                    '{:0>5}.2.enc{}.png'.format(_, FLAGS.postfix)) for _ in _range])
                helper.WriteFiles(cur_pngs, ofiles)
            
            # summary
            print('No.{}'.format(FLAGS.postfix))
            mean_loss = [l / max_steps for l in sum_loss]
            psnr = 10 * np.log10(1 / mean_loss[0]) if mean_loss[0] > 0 else 100
            print('PSNR (RGB) {}, MAD (RGB) {}, SS-SSIM(Y) {}, MS-SSIM (Y) {}, Sparsity {}'.format(
                   psnr, *mean_loss[1:]))
        
        # test progressively saved models
        if FLAGS.progress:
            mfiles = helper.listdir_files(FLAGS.train_dir, recursive=False,
                                          filter_ext=['.index'],
                                          encoding=None)
            mfiles = [f[:-6] for f in mfiles if 'model_' in f]
            mfiles.sort()
            stats = []
        else:
            mfiles = []
        
        for model_file in mfiles:
            with setup() as sess:
                # restore variables from saved model
                saver.restore(sess, model_file)
                
                # run session
                sum_loss = [0 for _ in range(len(ret_loss))]
                for step in range(max_steps):
                    cur_loss = sess.run(ret_loss)
                    # monitor losses
                    for _ in range(len(ret_loss)):
                        sum_loss[_] += cur_loss[_]
                
                # summary
                mean_loss = [l / max_steps for l in sum_loss]
                
                # save stats
                if FLAGS.progress:
                    model_num = os.path.split(model_file)[1][6:]
                    stats.append(np.array([float(model_num)] + mean_loss))
    
    # save stats
    import matplotlib.pyplot as plt
    if FLAGS.progress:
        stats = np.stack(stats)
        np.save(os.path.join(FLAGS.test_dir, 'stats.npy'), stats)
        fig, ax = plt.subplots()
        ax.set_title('Test Error with Training Progress')
        ax.set_xlabel('training steps')
        ax.set_ylabel('MAD (RGB)')
        ax.set_xscale('linear')
        ax.set_yscale('log')
        stats = stats[1:]
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
