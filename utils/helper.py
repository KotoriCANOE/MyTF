import tensorflow as tf

# divide 2 integers and round up
def divUp(dividend, divisor):
    return (dividend + divisor - 1) // divisor

# make directory
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
    for (dirpath, dirnames, filenames) in os.walk(path):
        for f in filenames:
            if os.path.splitext(f)[1].lower() in filter_ext:
                file_path = '{}/{}'.format(dirpath, f)
                if encoding: file_path = file_path.encode(encoding)
                files.append(file_path)
        if not recursive: break
    return files

# reading images using FIFOQueue within tensorflow graph
def ImageReader(files, channels=0, shuffle=False):
    file_queue = tf.train.string_input_producer(files, shuffle=shuffle)
    reader = tf.WholeFileReader()
    key, value = reader.read(file_queue)
    image = tf.image.decode_image(value, channels=channels)
    image.set_shape([None, None, 3])
    return image

# writing batch of images within tensorflow graph
def ImageBatchWriter(sess, images, files, dtype=tf.uint8):
    pngs = []
    for i in range(len(files)):
        img = images[i]
        img = tf.image.convert_image_dtype(img, dtype, saturate=True)
        png = tf.image.encode_png(img, compression=9)
        pngs.append(png)
    pngs = sess.run(pngs)
    for i in range(len(files)):
        with open(files[i], 'wb') as f:
            f.write(pngs[i])
