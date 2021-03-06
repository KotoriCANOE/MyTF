import tensorflow as tf
import numpy as np

# stderr print
def eprint(*args, **kwargs):
    import sys
    print(*args, file=sys.stderr, **kwargs)

# divide 2 integers and round up
def DivUp(dividend, divisor):
    return (dividend + divisor - 1) // divisor

# make directory
def make_dirs(path):
    import locale
    encoding = locale.getpreferredencoding()
    path = path.encode(encoding)
    if not tf.gfile.Exists(path):
        tf.gfile.MakeDirs(path)

def mkdir(path):
    import os
    # remove front blanks
    path = path.strip()
    # remove / and \ at the end
    path = path.rstrip('/')
    path = path.rstrip('\\')

    if os.path.exists(path):
        return False
    else:
        os.makedirs(path)
        return True

# recursively list all the files' path under directory
def listdir_files(path, recursive=True, filter_ext=None, encoding=None):
    import os, locale
    if encoding is True: encoding = locale.getpreferredencoding()
    if filter_ext is not None: filter_ext = [e.lower() for e in filter_ext]
    files = []
    for (dir_path, dir_names, file_names) in os.walk(path):
        for f in file_names:
            if not filter_ext or os.path.splitext(f)[1].lower() in filter_ext:
                file_path = os.path.join(dir_path, f)
                try:
                    if encoding: file_path = file_path.encode(encoding)
                    files.append(file_path)
                except UnicodeEncodeError as err:
                    eprint(file_path)
                    eprint(err)
        if not recursive: break
    return files

# convert list of tf.Dimension to list of int/None
def dim2int(shape):
    return [s.value if isinstance(s, tf.Dimension) else s for s in shape]

# get Session from MoniteredSession
def get_session(mon_sess):
    session = mon_sess
    while type(session).__name__ != 'Session':
        #pylint: disable=W0212
        session = session._sess
    return session

# return the string length(s) in a Tensor of Strings
def string_length(t):
    def _func(p):
        is_array = isinstance(p, np.ndarray)
        if is_array:
            return np.stack([len(_) for _ in p])
        else:
            return len(p)
    l = tf.py_func(_func, [t], [tf.int32])
    return l[0]

# reading images using FIFOQueue within tensorflow graph
def ImageReader(files, channels=0, shuffle=False):
    file_queue = tf.train.string_input_producer(files, shuffle=shuffle)
    reader = tf.WholeFileReader()
    key, value = reader.read(file_queue)
    image = tf.image.decode_image(value, channels=channels)
    #image.set_shape([None, None, None if channels <= 0 else channels])
    return image

# encode a batch of images to a list of pngs
def BatchPNG(images, batch_size, dtype=tf.uint8):
    pngs = []
    for i in range(batch_size):
        img = images[i]
        if img.dtype != tf.uint8: img = tf.image.convert_image_dtype(img, dtype, saturate=True)
        png = tf.image.encode_png(img, compression=9)
        pngs.append(png)
    return pngs

def WriteFiles(bytes, files):
    for i in range(len(files)):
        with open(files[i], 'wb') as f:
            f.write(bytes[i])

def ImageBatchWriter(sess, images, files, dtype=tf.uint8):
    pngs = BatchPNG(images, len(files))
    pngs = sess.run(pngs)
    WriteFiles(pngs, files)

# Gaussian filter window for Conv2D
def gauss_window(radius, sigma, channels=1, dtype=tf.float32):
    # w = exp((x*x + y*y) / (-2.0*sigma*sigma))
    x_data, y_data = np.mgrid[-radius:1+radius, -radius:1+radius]
    w_data = (np.square(x_data) + np.square(y_data)) * -0.5
    w_data = np.expand_dims(w_data, axis=-1)
    w_data = np.expand_dims(w_data, axis=-1)
    
    w = tf.constant(w_data, dtype=dtype)
    if not isinstance(sigma, tf.Tensor):
        sigma = tf.constant(sigma, dtype=dtype)
    g = tf.exp(w / tf.square(sigma))
    g /= tf.reduce_sum(g)
    
    if channels > 1:
        g = tf.concat([g] * channels, axis=-2)
    return g
