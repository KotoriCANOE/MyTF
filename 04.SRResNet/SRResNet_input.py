import sys
import tensorflow as tf

# flags
FLAGS = tf.app.flags.FLAGS

def inputs(files, is_training=False, is_testing=False):
    # parameters
    channels = FLAGS.image_channels
    threads = FLAGS.threads
    scaling = FLAGS.scaling
    if is_training: num_epochs = FLAGS.num_epochs
    patch_height = FLAGS.patch_height
    patch_width = FLAGS.patch_width
    batch_size = FLAGS.batch_size
    if is_training: buffer_size = FLAGS.buffer_size

    # dataset mapping function
    def parse1_func(filename):
        # read data
        dtype = tf.float32
        image = tf.read_file(filename)
        image = tf.image.decode_image(image, channels=channels)
        shape = tf.shape(image)
        height = shape[-3]
        width = shape[-2]
        # pre down-scale for high resolution image
        dscale = 1
        if is_training:
            '''
            if (width >= 3072 and height >= 1536) or (width >= 1536 and height >= 3072):
                dscale = 3
            elif (width >= 1024 and height >= 512) or (width >= 512 and height >= 1024):
                dscale = 2
            '''
            def c_t(const1, const2, true_fn, false_fn):
                return tf.cond(tf.logical_or(
                    tf.logical_and(
                        tf.greater_equal(width, const1), tf.greater_equal(height, const2)
                    ),
                    tf.logical_and(
                        tf.greater_equal(width, const2), tf.greater_equal(height, const1)
                    )
                ), true_fn, false_fn)
            dscale = c_t(3072, 1536, lambda: 3,
                lambda: c_t(1024, 512, lambda: 2, lambda: 1)
            )
        elif is_testing:
            '''
            if (width >= 3072 and height >= 3072):
                dscale = 4
            elif (width >= 2048 and height >= 2048):
                dscale = 3
            elif (width >= 1024 and height >= 1024):
                dscale = 2
            '''
            def c_t(const1, true_fn, false_fn):
                return tf.cond(tf.logical_and(
                    tf.greater_equal(width, const1), tf.greater_equal(height, const1)
                ), true_fn, false_fn)
            dscale = c_t(3072, lambda: 4,
                lambda: c_t(2048, lambda: 3,
                    lambda: c_t(1024, lambda: 2, lambda: 1)
                )
            )
        # padding
        cropped_height = patch_height * dscale
        cropped_width = patch_width * dscale
        '''
        if cropped_height > height or cropped_width > width:
            pad_height = cropped_height - height
            pad_width = cropped_width - width
            if pad_height > 0:
                pad_height = [pad_height // 2, pad_height - pad_height // 2]
                height = cropped_height
            else:
                pad_height = [0, 0]
            if pad_width > 0:
                pad_width = [pad_width // 2, pad_width - pad_width // 2]
                width = cropped_width
            else:
                pad_width = [0, 0]
            block = tf.pad(image, [pad_height, pad_width, [0, 0]], mode='REFLECT')
        else:
            block = image
        '''
        cond_height = tf.greater(cropped_height, height)
        cond_width = tf.greater(cropped_width, width)
        def c_f1():
            def _1():
                ph = cropped_height - height
                return [ph // 2, ph - ph // 2]
            pad_height = tf.cond(cond_height, _1, lambda: [0, 0])
            def _2():
                pw = cropped_width - width
                return [pw // 2, pw - pw // 2]
            pad_width = tf.cond(cond_width, _2, lambda: [0, 0])
            return tf.pad(image, [pad_height, pad_width, [0, 0]], mode='REFLECT')
        block = tf.cond(tf.logical_or(cond_height, cond_width), c_f1, lambda: image)
        height = tf.maximum(cropped_height, height)
        width = tf.maximum(cropped_width, width)
        # cropping
        if is_training:
            block = tf.random_crop(block, [cropped_height, cropped_width, channels])
            block = tf.image.random_flip_up_down(block)
            block = tf.image.random_flip_left_right(block)
        elif is_testing:
            offset_height = (height - cropped_height) // 2
            offset_width = (width - cropped_width) // 2
            block = tf.image.crop_to_bounding_box(block, offset_height, offset_width,
                                                  cropped_height, cropped_width)
        block = tf.image.convert_image_dtype(block, dtype, saturate=False)
        if is_training and FLAGS.color_augmentation > 0:
            block = tf.image.random_saturation(block, 1 - FLAGS.color_augmentation, 1 + FLAGS.color_augmentation)
            block = tf.image.random_brightness(block, FLAGS.color_augmentation)
            block = tf.image.random_contrast(block, 1 - FLAGS.color_augmentation, 1 + FLAGS.color_augmentation)
        # process 2
        data_size = [patch_height // scaling, patch_width // scaling]
        data = tf.image.resize_images(block, data_size, tf.image.ResizeMethod.AREA)
        '''
        if dscale != 1:
            label_size = [patch_height, patch_width]
            label = tf.image.resize_images(block, label_size, tf.image.ResizeMethod.AREA)
        else:
            label = block
        '''
        def c_f1():
            label_size = [patch_height, patch_width]
            return tf.image.resize_images(block, label_size, tf.image.ResizeMethod.AREA)
        label = tf.cond(tf.not_equal(dscale, 1), c_f1, lambda: block)
        # process 3
        data_shape = [patch_height // scaling, patch_width // scaling, channels]
        if is_training and FLAGS.noise_level > 0:
            noise = tf.random_normal(data_shape, mean=0.0, stddev=FLAGS.noise_level, dtype=data.dtype)
            data = tf.add(data, noise)
        data = tf.clip_by_value(data, 0.0, 1.0)
        label = tf.clip_by_value(label, 0.0, 1.0)
        # data format conversion
        data.set_shape([patch_height // scaling, patch_width // scaling, channels])
        label.set_shape([patch_height, patch_width, channels])
        if FLAGS.data_format == 'NCHW':
            data = tf.transpose(data, (2, 0, 1))
            label = tf.transpose(label, (2, 0, 1))
        # return
        return data, label

    # Dataset API
    dataset = tf.contrib.data.Dataset.from_tensor_slices(tf.constant(files))
    dataset = dataset.map(parse1_func, num_threads=threads,
                          output_buffer_size=threads * 64)
    if is_training: dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size)
    if is_training: dataset = dataset.repeat(num_epochs)

    # return iterator
    iterator = dataset.make_one_shot_iterator()
    next_data, next_label = iterator.get_next()
    
    return next_data, next_label
