import os
import time
import queue
import threading
import numpy as np
import tensorflow as tf

# working directory
print('Current working directory:\n    {}\n'.format(os.getcwd()))
MODEL_DIR = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'SR_model')

# flags
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('model_dir', MODEL_DIR,
                           """Directory where the image files are to be processed.""")
tf.app.flags.DEFINE_integer('scaling', 2,
                            """Up-scaling factor of the model.""")
tf.app.flags.DEFINE_string('src_dir', './',
                           """Directory where the image files are to be processed.""")
tf.app.flags.DEFINE_string('dst_dir', os.path.join(FLAGS.src_dir, 'SR_results'),
                           """Directory where to write the processed images.""")
tf.app.flags.DEFINE_string('dst_postfix', '.SR',
                           """Postfix added to the processed filenames.""")
tf.app.flags.DEFINE_boolean('recursive', True,
                           """Recursively search all the files in 'src_dir'.""")
tf.app.flags.DEFINE_string('data_format', 'NCHW', # 'NHWC'
                            """Data layout format.""")
tf.app.flags.DEFINE_integer('patch_height', 512,
                            """Max patch height.""")
tf.app.flags.DEFINE_integer('patch_width', 512,
                            """Max patch width.""")
tf.app.flags.DEFINE_integer('patch_pad', 8,
                            """Padding around patches.""")
tf.app.flags.DEFINE_integer('threads', 0,
                            """Concurrent multi-threading Python execution.""")
tf.app.flags.DEFINE_integer('sess_threads', 1,
                            """Maximum number of concurrent running TensorFlow sessions.""")
tf.app.flags.DEFINE_float('memory_fraction', 1.0,
                            """Maximum allowed fraction of memory to allocate.""")
tf.app.flags.DEFINE_string('device', 'GPU:0',
                            """Preferred device to use.""")

# stderr print
def eprint(*args, **kwargs):
    import sys
    print(*args, file=sys.stderr, **kwargs)

# API
class SRFilter:
    def __init__(self, model_dir=MODEL_DIR, data_format='NCHW', scaling=2,
                 sess_threads=1, memory_fraction=1.0, device='GPU:0'):
        # arXiv 1509.09308
        # a new class of fast algorithms for convolutional neural networks using Winograd's minimal filtering algorithms
        os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
        
        self.data_format = data_format
        self.scaling = scaling
        self.memory_fraction = memory_fraction
        self.device = '/device:{}'.format(device)
        
        self._create_graph()
        self._create_session()
        self._restore_graph(model_dir)
        self.semaphore = threading.Semaphore(value=sess_threads)
    
    def _create_graph(self):
        self.graph = tf.Graph()
    
    def _create_session(self):
        gpu_options = tf.GPUOptions(allow_growth=True,
            per_process_gpu_memory_fraction=self.memory_fraction)
        config = tf.ConfigProto(gpu_options=gpu_options,
            allow_soft_placement=True, log_device_placement=False)
        self.sess = tf.Session(graph=self.graph, config=config)
    
    def _restore_graph(self, model_dir):
        # force loading contrib ops
        # https://github.com/tensorflow/tensorflow/issues/10130
        dir(tf.contrib)
        # load meta graph and restore variables
        with self.graph.as_default():
            with tf.device(self.device):
                saver = tf.train.import_meta_graph(os.path.join(model_dir, 'model.meta'),
                    clear_devices=True)
                if saver is None:
                    raise ValueError('Failed to import meta graph!')
                saver.restore(self.sess, os.path.join(model_dir, 'model'))
        # access placeholders variables
        self.input = self.graph.get_tensor_by_name('Input:0')
        self.output = self.graph.get_tensor_by_name('Output:0')
        self.graph.finalize()
    
    def inference(self, input):
        feed_dict = {self.input: input}
        with self.semaphore:
            output = self.sess.run(self.output, feed_dict)
        return output
    
    def process(self, src, max_patch_height=360, max_patch_width=360, patch_pad=8, data_format='NHWC', msg=None):
        assert isinstance(src, np.ndarray)
        if msg is True: msg = ''
        # shape standardization
        src_shape = src.shape
        if len(src_shape) == 2:
            src = np.expand_dims(src, 0)
            src = np.expand_dims(src, 0)
            data_format = 'NCHW'
        elif len(src_shape) == 3:
            src = np.expand_dims(src, 0)
        elif len(src_shape) != 4:
            raise ValueError('Not supported rank of \'src\', should be 4, 3 or 2.')
        if data_format != 'NCHW':
            src = src.transpose((0, 3, 1, 2))
        src_channels = src.shape[1]
        if src_channels == 1:
            src = np.concatenate([src, src, src], axis=1)
        elif src_channels != 3:
            raise ValueError('Only 3 channels or 1 channel is supported, but \'src\' has {} channels.\n'
                             'If it\'s an image with alpha channel, '
                             'you should process the alpha channel separately using other resizers.'
                             .format(src_channels))
        # convert to float32
        src_dtype = src.dtype
        if src_dtype != np.float32:
            src = src.astype(np.float32)
            if src_dtype == np.uint8:
                src *= 1 / 255
            elif src_dtype == np.uint16:
                src *= 1 / 65535
        # parameters
        shape = src.shape
        height = shape[2]
        width = shape[3]
        # split into (padded & overlapped) patches
        def pad_split_patch(dim, max_patch, patch_pad=0):
            split = 1
            patch = (dim + patch_pad * split * 2 + split - 1) // split
            while patch > max_patch:
                split += 1
                patch = (dim + patch_pad * split * 2 + split - 1) // split
            pad = patch * split - patch_pad * (split - 1) * 2 - dim
            return pad, split, patch
        pad_h, split_h, patch_h = pad_split_patch(height, max_patch_height, patch_pad)
        pad_w, split_w, patch_w = pad_split_patch(width, max_patch_width, patch_pad)
        # padding
        need_padding = pad_h > 0 or pad_w > 0
        pad_h = (patch_pad, pad_h - patch_pad)
        pad_w = (patch_pad, pad_w - patch_pad)
        if need_padding:
            src = np.pad(src, ((0, 0), (0, 0), pad_h, pad_w), mode='reflect')
        # splitting
        splits = split_h * split_w
        src_patches = []
        for s in range(splits):
            p_h = (s // split_w) * (patch_h - patch_pad * 2)
            p_w = (s % split_w) * (patch_w - patch_pad * 2)
            src_patches.append(src[:, :, p_h : p_h + patch_h, p_w : p_w + patch_w])
        # inference
        if msg or msg == '':
            eprint(msg + 'Inferencing using model...')
            _t = time.time()
        dst_patches = []
        for src_p in src_patches:
            if self.data_format != 'NCHW':
                src_p = src_p.transpose((0, 2, 3, 1))
            dst_p = self.inference(src_p)
            if self.data_format != 'NCHW':
                dst_p = dst_p.transpose((0, 3, 1, 2))
            dst_patches.append(dst_p)
        if msg or msg == '':
            _d = time.time() - _t
            eprint(msg + 'Inferencing finished. Duration: {} seconds.'.format(_d))
        # cropping
        for s in range(splits):
            crop_t = (pad_h[0] if s // split_w == 0 else patch_pad) * self.scaling
            crop_b = (pad_h[1] if s // split_w == split_h - 1 else patch_pad) * self.scaling
            crop_l = (pad_w[0] if s % split_w == 0 else patch_pad) * self.scaling
            crop_r = (pad_w[1] if s % split_w == split_w - 1 else patch_pad) * self.scaling
            if crop_t > 0 or crop_b > 0 or crop_l > 0 or crop_r > 0:
                crop_t = None if crop_t <= 0 else crop_t
                crop_b = None if crop_b <= 0 else -crop_b
                crop_l = None if crop_l <= 0 else crop_l
                crop_r = None if crop_r <= 0 else -crop_r
                dst_patches[s] = dst_patches[s][:, :, crop_t:crop_b, crop_l:crop_r]
        # stacking (concatenating)
        dst_patches_h = []
        for s_h in range(split_h):
            s = s_h * split_w
            dst_patches_h.append(np.concatenate(dst_patches[s : s + split_w], axis=-1))
        dst = np.concatenate(dst_patches_h, axis=-2)
        # clipping output value
        dst = np.clip(dst, 0, 1)
        # convert to src_type
        if src_dtype != np.float32:
            if src_dtype == np.uint8:
                dst *= 255
            elif src_dtype == np.uint16:
                dst *= 65535
            dst = dst.astype(src_dtype)
        # reshape to src_shape
        if src_channels == 1:
            dst = dst[:, :1, :, :]
        if len(src_shape) == 2:
            dst = np.squeeze(dst, (0, 1))
            data_format = 'NCHW'
        if data_format != 'NCHW':
            dst = dst.transpose((0, 2, 3, 1))
        if len(src_shape) == 3:
            dst = np.squeeze(dst, 0)
        # return
        return dst

# recursively list all the files' path under directory
def listdir_files(path, recursive=True, filter_ext=None, encoding=None):
    import os, locale
    if encoding is True: encoding = locale.getpreferredencoding()
    if filter_ext is not None: filter_ext = [e.lower() for e in filter_ext]
    files = []
    for (dir_path, dir_names, file_names) in os.walk(path):
        for f in file_names:
            if os.path.splitext(f)[1].lower() in filter_ext:
                file_path = os.path.join(dir_path, f)
                if encoding: file_path = file_path.encode(encoding)
                files.append((file_path, dir_path, os.path.splitext(f)[0]))
        if not recursive: break
    return files

# main
def main(argv=None):
    from skimage import io
    from skimage import transform
    
    extensions = ['.jpeg', '.jpg', '.png', '.bmp', '.webp', '.tif', '.tiff', '.jp2']
    dst_postfix = FLAGS.dst_postfix + '.png'
    channel_index = -1
    # nunber of threads
    if FLAGS.threads <= 0:
        thread_num = max(1, os.cpu_count() - FLAGS.threads)
    else:
        thread_num = FLAGS.threads
    # directories and files
    if not os.path.exists(FLAGS.dst_dir): os.makedirs(FLAGS.dst_dir)
    src_files = listdir_files(FLAGS.src_dir, FLAGS.recursive, extensions)
    # initialization
    filter = SRFilter(FLAGS.model_dir, FLAGS.data_format, FLAGS.scaling,
        FLAGS.sess_threads, FLAGS.memory_fraction, FLAGS.device)
    # worker - read, process and save image files
    def worker(q, t):
        msg = '{}: '.format(t) if thread_num > 1 else ''
        while True:
            # dequeue
            item = q.get()
            if item is None:
                break
            else:
                file_path, dir_path, file_name = item
            # read
            src = io.imread(file_path)
            print(msg + 'Loaded {}'.format(file_path))
            # separate alpha channel
            channels = src.shape[channel_index]
            if channels == 2 or channels == 4:
                alpha = src[:, :, -1:]
                src = src[:, :, :-1]
            # process
            dst = filter.process(src, max_patch_height=FLAGS.patch_height, max_patch_width=FLAGS.patch_width,
                                 patch_pad=FLAGS.patch_pad, data_format='NHWC', msg=None if thread_num > 1 else True)
            # process and merge alpha channel
            if channels == 2 or channels == 4:
                dtype = alpha.dtype
                bits = dtype.itemsize * 8
                alpha = transform.rescale(alpha, FLAGS.scaling, mode='edge', preserve_range=True)
                if bits < 32:
                    alpha = np.clip(alpha, 0, (1 << bits) - 1)
                    alpha = alpha.astype(dtype)
                else:
                    alpha = np.clip(alpha, 0, 1)
                dst = np.concatenate([dst, alpha], axis=channel_index)
            # save
            save_dir = dir_path[len(FLAGS.src_dir):].strip('/').strip('\\')
            save_dir = os.path.join(FLAGS.dst_dir, save_dir)
            if not os.path.exists(save_dir): os.makedirs(save_dir)
            save_file = os.path.join(save_dir, file_name + dst_postfix)
            if thread_num == 1: print(msg + 'Saving... {}'.format(save_file))
            io.imsave(save_file, dst)
            print(msg + 'Result saved to {}'.format(save_file))
            # indicate enqueued task is complete
            q.task_done()
    # enqueue
    q = queue.Queue()
    for item in src_files:
        q.put(item)
    # start threads
    threads = []
    for _ in range(thread_num):
        t = threading.Thread(target=lambda: worker(q, _))
        t.start()
        threads.append(t)
    # block until all tasks are done
    q.join()
    # stop workers
    for i in range(thread_num):
        q.put(None)
    for t in threads:
        t.join()

# tf main
if __name__ == '__main__':
    tf.app.run()
