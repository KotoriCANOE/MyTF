import sys
import os
import numpy as np
import tensorflow as tf
sys.path.append('..')
from utils import helper
import utils.image

from input import inputs
from model import SRmodel

# working directory
print('Current working directory:\n    {}\n'.format(os.getcwd()))

# flags
FLAGS = tf.app.flags.FLAGS

# parameters
tf.app.flags.DEFINE_string('postfix', '',
                            """Postfix added to train_dir, test_dir, test files, etc.""")
tf.app.flags.DEFINE_string('train_dir', './train{postfix}.tmp',
                           """Directory where to read checkpoint.""")
tf.app.flags.DEFINE_string('test_dir', './test{postfix}.tmp',
                           """Directory where to write event logs and test results.""")
tf.app.flags.DEFINE_string('dataset', '../../Dataset.SR/Test',
                           """Directory where stores the dataset.""")
tf.app.flags.DEFINE_string('device', '/gpu:0',
                            """Default device to place the graph.""")
tf.app.flags.DEFINE_boolean('progress', False,
                            """Whether to test across the entire training procedure.""")
tf.app.flags.DEFINE_integer('random_seed', 0,
                            """Initialize with specific random seed. Negative value to use system default.""")
tf.app.flags.DEFINE_integer('threads', 16,
                            """Number of threads for Dataset process.""")
tf.app.flags.DEFINE_integer('threads_py', 16,
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
tf.app.flags.DEFINE_integer('multistage_resize', 2,
                            """Apply resizer (n * 2 + 1) times to simulate more complex filtered data.""")
tf.app.flags.DEFINE_float('random_resizer', 0,
                            """value for resizer choice, 0 for random resizer.""")
tf.app.flags.DEFINE_float('noise_scale', 0.01,
                            """STD of additive Gaussian random noise.""")
tf.app.flags.DEFINE_float('noise_corr', 0.75,
                            """Spatial correlation of the Gaussian random noise.""")
tf.app.flags.DEFINE_float('jpeg_coding', 2.0,
                            """Using JPEG to generate compression artifacts for data.""")

# reset random seeds
def reset_random(seed=0):
    import random
    tf.set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

# setup tensorflow and return session
def session():
    # initialize rng with a deterministic seed
    if FLAGS.random_seed >= 0:
        reset_random(FLAGS.random_seed)
    # create session
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options,
        allow_soft_placement=True, log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=config)
    return sess

# losses
def get_losses(ref, pred):
    # RGB color space
    RGB_mse = tf.losses.mean_squared_error(ref, pred, weights=1.0)
    RGB_mad = tf.losses.absolute_difference(ref, pred, weights=1.0)
    
    # OPP color space - Y
    Y_ref = utils.image.RGB2Y(ref, data_format=FLAGS.data_format)
    Y_pred = utils.image.RGB2Y(pred, data_format=FLAGS.data_format)
    Y_ss_ssim = utils.image.SS_SSIM(Y_ref, Y_pred, data_format=FLAGS.data_format)
    Y_ms_ssim = utils.image.MS_SSIM2(Y_ref, Y_pred, norm=True, data_format=FLAGS.data_format)
    Y_ms_ssim_unnorm = utils.image.MS_SSIM2(Y_ref, Y_pred, norm=False, data_format=FLAGS.data_format)
    
    # weighted loss
    alpha = 0.10
    loss = (1 - alpha) * RGB_mad + alpha * (1 - Y_ms_ssim_unnorm)
    
    # return each loss
    return RGB_mse, RGB_mad, Y_ss_ssim, Y_ms_ssim, loss

# testing
def test():
    # get dataset
    files = helper.listdir_files(FLAGS.dataset,
                                 filter_ext=['.jpeg', '.jpg', '.png'])
    steps_per_epoch = len(files) // FLAGS.batch_size
    epoch_size = steps_per_epoch * FLAGS.batch_size
    max_steps = steps_per_epoch
    files = files[:epoch_size]
    
    # test set
    test_lr_batches = []
    test_hr_batches = []
    with tf.Graph().as_default():
        # dataset
        with tf.device('/cpu:0'):
            test_lr, test_hr = inputs(FLAGS, files, is_testing=True)
        with session() as sess:
            for _ in range(max_steps):
                _lr, _hr = sess.run((test_lr, test_hr))
                test_lr_batches.append(_lr)
                test_hr_batches.append(_hr)
    
    with tf.Graph().as_default():
        images_lr = tf.placeholder(tf.float32, name='InputLR')
        images_hr = tf.placeholder(tf.float32, name='InputHR')
        
        # build model
        model = SRmodel(FLAGS, data_format=FLAGS.data_format,
            input_range=FLAGS.input_range, output_range=FLAGS.output_range,
            multiGPU=FLAGS.multiGPU, use_fp16=FLAGS.use_fp16,
            scaling=FLAGS.scaling, image_channels=FLAGS.image_channels)
        
        model.build_model(images_lr)
        
        # a Saver object to restore the variables with mappings
        saver = tf.train.Saver(var_list=model.g_rvars)
        
        # get output
        images_sr = tf.get_default_graph().get_tensor_by_name('Output:0')
        
        # losses
        ret_loss = list(get_losses(images_hr, images_sr))
        
        # post-processing for output
        with tf.device('/cpu:0'):
            # data format conversion
            if FLAGS.data_format == 'NCHW':
                images_lr = utils.image.NCHW2NHWC(images_lr)
                images_hr = utils.image.NCHW2NHWC(images_hr)
                images_sr = utils.image.NCHW2NHWC(images_sr)
            
            # Bicubic upscaling
            shape = tf.shape(images_lr)
            upsize = [shape[-3] * FLAGS.scaling, shape[-2] * FLAGS.scaling]
            images_bicubic = tf.image.resize_images(images_lr, upsize,
                    tf.image.ResizeMethod.BICUBIC, align_corners=True)
            
            # PNGs output
            ret_pngs = []
            ret_pngs.extend(helper.BatchPNG(images_lr, FLAGS.batch_size))
            ret_pngs.extend(helper.BatchPNG(images_hr, FLAGS.batch_size))
            ret_pngs.extend(helper.BatchPNG(images_sr, FLAGS.batch_size))
            ret_pngs.extend(helper.BatchPNG(images_bicubic, FLAGS.batch_size))
        
        # initialize session
        sess = session()
        
        # test latest checkpoint
        model_file = tf.train.latest_checkpoint(FLAGS.train_dir)
        saver.restore(sess, model_file)
        
        ret = ret_loss + ret_pngs
        sum_loss = [0 for _ in range(len(ret_loss))]
        for step in range(max_steps):
            feed_dict = {'InputLR:0': test_lr_batches[step], 'InputHR:0': test_hr_batches[step]}
            cur_ret = sess.run(ret, feed_dict=feed_dict)
            cur_loss = cur_ret[0:len(ret_loss)]
            cur_pngs = cur_ret[len(ret_loss):]
            # monitor losses
            for _ in range(len(ret_loss)):
                sum_loss[_] += cur_loss[_]
            #print('batch {}, MSE (RGB) {}, MAD (RGB) {},'
            #      'SS-SSIM(Y) {}, MS-SSIM (Y) {}, loss {}'
            #      .format(step, *cur_loss))
            # images output
            _start = step * FLAGS.batch_size
            _stop = _start + FLAGS.batch_size
            _range = range(_start, _stop)
            ofiles = []
            ofiles.extend([os.path.join(FLAGS.test_dir,
                '{:0>5}.0.LR.png'.format(_)) for _ in _range])
            ofiles.extend([os.path.join(FLAGS.test_dir,
                '{:0>5}.1.HR.png'.format(_)) for _ in _range])
            ofiles.extend([os.path.join(FLAGS.test_dir,
                '{:0>5}.2.SR{}.png'.format(_, FLAGS.postfix)) for _ in _range])
            ofiles.extend([os.path.join(FLAGS.test_dir,
                '{:0>5}.3.Bicubic.png'.format(_)) for _ in _range])
            helper.WriteFiles(cur_pngs, ofiles)
        
        # summary
        print('No.{}'.format(FLAGS.postfix))
        mean_loss = [l / max_steps for l in sum_loss]
        psnr = 10 * np.log10(1 / mean_loss[0]) if mean_loss[0] > 0 else 100
        print('PSNR (RGB) {}, MAD (RGB) {}, '
              'SS-SSIM(Y) {}, MS-SSIM (Y) {}, loss {}'
              .format(psnr, *mean_loss[1:]))
        
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
            # restore variables from saved model
            saver.restore(sess, model_file)
            
            # run session
            sum_loss = [0 for _ in range(len(ret_loss))]
            for step in range(max_steps):
                feed_dict = {'InputLR:0': test_lr_batches[step], 'InputHR:0': test_hr_batches[step]}
                cur_loss = sess.run(ret_loss, feed_dict=feed_dict)
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
        #ax.axis(ymin=0)
        plt.tight_layout()
        plt.savefig(os.path.join(FLAGS.test_dir, 'stats.png'))
        plt.close()
    
    print('')

# stderr print
def eprint(*args, **kwargs):
    import sys
    print(*args, file=sys.stderr, **kwargs)

# main
def main(argv=None):
    import shutil
    # arXiv 1509.09308
    # a new class of fast algorithms for convolutional neural networks using Winograd's minimal filtering algorithms
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    
    FLAGS.train_dir = FLAGS.train_dir.format(postfix=FLAGS.postfix)
    FLAGS.test_dir = FLAGS.test_dir.format(postfix=FLAGS.postfix)
    
    if not os.path.exists(FLAGS.train_dir):
        raise FileNotFoundError('Could not find folder {}'.format(FLAGS.train_dir))
    if os.path.exists(FLAGS.test_dir):
        eprint('Removed :' + FLAGS.test_dir)
        shutil.rmtree(FLAGS.test_dir)
    os.makedirs(FLAGS.test_dir)
    
    with tf.device(FLAGS.device):
        test()

if __name__ == '__main__':
    tf.app.run()
