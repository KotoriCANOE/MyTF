import sys
import tensorflow as tf
import vapoursynth as vs
sys.path.append('..')
from utils import vshelper

# flags
FLAGS = tf.app.flags.FLAGS

def inputs(files):
    # parameters
    channels = 3
    threads = FLAGS.threads
    scaling = FLAGS.scaling
    num_epochs = FLAGS.num_epochs
    block_size = FLAGS.block_size
    batch_size = FLAGS.batch_size
    buffer_size = FLAGS.buffer_size

    # vapoursynth
    core = vs.get_core(threads=1)
    core.max_cache_size = 2000

    # dataset mapping function
    def parse1_func(filename):
        # read data
        dtype = tf.float32
        image = tf.read_file(filename)
        image = tf.image.decode_image(image, channels=channels)
        image.set_shape([None, None, channels])
        shape = image.get_shape()
        height = shape[-3]
        width = shape[-2]
        if (width >= 3072 and height >= 1536) or (width >= 1536 and height >= 3072):
            dscale = 4
        elif (width >= 2048 and height >= 1024) or (width >= 1024 and height >= 2048):
            dscale = 3
        elif (width >= 1024 and height >= 512) or (width >= 512 and height >= 1024):
            dscale = 2
        else:
            dscale = 1
        # process 1
        block = tf.random_crop(image, [block_size * dscale, block_size * dscale, channels])
        block = tf.image.random_flip_up_down(block)
        block = tf.image.random_flip_left_right(block)
        block = tf.image.convert_image_dtype(block, dtype, saturate=False)
        block = tf.image.random_saturation(block, .95, 1.05)
        block = tf.image.random_brightness(block, .05)
        block = tf.image.random_contrast(block, .95, 1.05)
        # process 2
        data_size = [block_size // scaling, block_size // scaling]
        data = tf.image.resize_images(block, data_size, tf.image.ResizeMethod.AREA)
        if dscale != 1:
            label_size = [block_size, block_size]
            label = tf.image.resize_images(block, label_size, tf.image.ResizeMethod.AREA)
        else:
            label = block
        # process 3
        data_shape = [block_size // scaling, block_size // scaling, channels]
        noise = tf.random_normal(data_shape, mean=0.0, stddev=0.005, dtype=data.dtype)
        data = tf.add(data, noise)
        data = tf.clip_by_value(data, 0.0, 1.0)
        label = tf.clip_by_value(label, 0.0, 1.0)
        # return
        return data, label

    # Dataset API
    epoch_size = len(files) - len(files) % batch_size
    files = files[:epoch_size]
    dataset = tf.contrib.data.Dataset.from_tensor_slices(tf.constant(files))
    dataset = dataset.map(parse1_func, num_threads=threads,
                          output_buffer_size=threads * 64)
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)

    # return iterator
    iterator = dataset.make_one_shot_iterator()
    next_data, next_label = iterator.get_next()
    next_data.set_shape([batch_size, block_size // scaling, block_size // scaling, channels])
    next_label.set_shape([batch_size, block_size, block_size, channels])
    
    return next_data, next_label
