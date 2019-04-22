import os
import time
import queue
import threading
import numpy as np
import tensorflow as tf

# flags
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('src_dir', './',
                           """Directory where the image files are to be processed.""")
tf.app.flags.DEFINE_string('dst_dir', '{src_dir}/split',
                           """Directory where to write the processed images.""")
tf.app.flags.DEFINE_boolean('recursive', True,
                           """Recursively search all the files in 'src_dir'.""")
tf.app.flags.DEFINE_integer('patch_height', 1024,
                            """Max patch height.""")
tf.app.flags.DEFINE_integer('patch_width', 1024,
                            """Max patch width.""")
tf.app.flags.DEFINE_integer('patch_pad', 0,
                            """Padding around patches.""")
tf.app.flags.DEFINE_integer('threads', 0,
                            """Concurrent multi-threading Python execution.""")

# process - split
def process(src, max_patch_height=480, max_patch_width=480, patch_pad=48):
    assert isinstance(src, np.ndarray)
    # shape check
    src_shape = src.shape
    if len(src_shape) < 2:
        raise ValueError('Not supported rank of \'src\', should be at least 2.')
    # parameters
    shape = src.shape
    height = shape[0]
    width = shape[1]
    # split into (cropped & overlapped) patches
    def crop_split_patch(dim, max_patch, patch_pad=0):
        split = 1
        patch = (dim + patch_pad * (split - 1) * 2) // split
        while patch > max_patch:
            split += 1
            patch = (dim + patch_pad * (split - 1) * 2) // split
        crop = dim + patch_pad * (split - 1) * 2 - patch * split
        return crop, split, patch
    crop_h, split_h, patch_h = crop_split_patch(height, max_patch_height, patch_pad)
    crop_w, split_w, patch_w = crop_split_patch(width, max_patch_width, patch_pad)
    # cropping
    src = src[crop_h // 2 : height - crop_h // 2, crop_w // 2 : width - crop_w // 2]
    # splitting
    splits = split_h * split_w
    src_patches = []
    for s in range(splits):
        p_h = (s // split_w) * (patch_h - patch_pad * 2)
        if s // split_w > 0: p_h += patch_pad
        p_w = (s % split_w) * (patch_w - patch_pad * 2)
        if s % split_w > 0: p_w += patch_pad
        src_patches.append(src[p_h : p_h + patch_h, p_w : p_w + patch_w])
    # return
    return src_patches

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
    FLAGS.dst_dir = FLAGS.dst_dir.format(src_dir=FLAGS.src_dir)

    from skimage import io
    from skimage import transform
    
    extensions = ['.jpeg', '.jpg', '.png', '.bmp', '.webp', '.tif', '.tiff', '.jp2']
    channel_index = -1
    # nunber of threads
    if FLAGS.threads <= 0:
        thread_num = max(1, os.cpu_count() - FLAGS.threads)
    else:
        thread_num = FLAGS.threads
    # directories and files
    if not os.path.exists(FLAGS.dst_dir): os.makedirs(FLAGS.dst_dir)
    src_files = listdir_files(FLAGS.src_dir, FLAGS.recursive, extensions)
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
            # process
            dst = process(src, max_patch_height=FLAGS.patch_height, max_patch_width=FLAGS.patch_width,
                          patch_pad=FLAGS.patch_pad)
            # save
            save_dir = dir_path[len(FLAGS.src_dir):].strip('/').strip('\\')
            save_dir = os.path.join(FLAGS.dst_dir, save_dir)
            if not os.path.exists(save_dir): os.makedirs(save_dir)
            for n in range(len(dst)):
                file_postfix = '.{:0>3}.png'.format(n + 1)
                save_file = os.path.join(save_dir, file_name + file_postfix)
                if thread_num == 1: print(msg + 'Saving... {}'.format(save_file))
                io.imsave(save_file, dst[n])
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
