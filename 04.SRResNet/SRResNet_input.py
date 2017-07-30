import sys
import tensorflow as tf
import numpy as np

# flags
FLAGS = tf.app.flags.FLAGS

def inputs(files, is_training=False, is_testing=False):
    # parameters
    channels = FLAGS.image_channels
    threads = FLAGS.threads
    threads_py = FLAGS.threads if not is_testing else 1
    scaling = FLAGS.scaling
    if is_training: num_epochs = FLAGS.num_epochs
    data_format = FLAGS.data_format
    patch_height = FLAGS.patch_height
    patch_width = FLAGS.patch_width
    batch_size = FLAGS.batch_size
    if is_training: buffer_size = FLAGS.buffer_size
    epoch_size = len(files)

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
        if is_training and FLAGS.pre_down:
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
        elif is_testing and FLAGS.pre_down:
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
        # convert dtype
        block = tf.image.convert_image_dtype(block, dtype, saturate=False)
        # pre-downscale
        def c_f1():
            label_size = [patch_height, patch_width]
            return tf.image.resize_images(block, label_size, tf.image.ResizeMethod.AREA)
        block = tf.cond(tf.not_equal(dscale, 1), c_f1, lambda: block)
        # random color augmentation
        if is_training and FLAGS.color_augmentation > 0:
            block = tf.image.random_saturation(block, 1 - FLAGS.color_augmentation, 1 + FLAGS.color_augmentation)
            block = tf.image.random_brightness(block, FLAGS.color_augmentation)
            block = tf.image.random_contrast(block, 1 - FLAGS.color_augmentation, 1 + FLAGS.color_augmentation)
        # data format conversion
        block.set_shape([None, None, channels])
        if data_format == 'NCHW':
            block = tf.transpose(block, (2, 0, 1))
        # return
        return block
    
    # tf.py_func processing using vapoursynth, numpy, etc.
    import threading
    import vapoursynth as vs
    from scipy import ndimage
    
    _lock = threading.Lock()
    _index_ref = [0]
    _src_ref = [None for _ in range(epoch_size)]
    core = vs.get_core(threads=threads_py)
    _src_blk = core.std.BlankClip(None, patch_width, patch_height,
                                 format=vs.RGBS, length=epoch_size)
    _dst_blk = core.std.BlankClip(None, patch_width // scaling, patch_height // scaling,
                                 format=vs.RGBS, length=epoch_size)
    
    def src_func(n, f):
        f_out = f.copy()
        planes = f_out.format.num_planes
        # output
        for p in range(planes):
            f_arr = np.array(f_out.get_write_array(p), copy=False)
            np.copyto(f_arr, _src_ref[n][p, :, :] if data_format == 'NCHW' else _src_ref[n][:, :, p])
        return f_out
    _src = _src_blk.std.ModifyFrame(_src_blk, src_func)
    
    def eval_func(n):
        clip = _src
        rand_val = np.random.uniform(-1, 1)
        abs_rand = np.abs(rand_val)
        dw = patch_width // scaling
        dh = patch_height // scaling
        # random gamma-to-linear
        if rand_val < 0: clip = clip.resize.Bicubic(transfer_s='linear', transfer_in_s='709')
        # random resizers
        if abs_rand < 0.05:
            clip = clip.resize.Bilinear(dw, dh)
        elif abs_rand < 0.1:
            clip = clip.resize.Spline16(dw, dh)
        elif abs_rand < 0.15:
            clip = clip.resize.Spline36(dw, dh)
        elif abs_rand < 0.3: # Lanczos taps=[3, 24)
            taps = int(np.clip(np.random.exponential(3), 0, 20)) + 3
            clip = clip.resize.Lanczos(dw, dh, filter_param_a=taps)
        elif abs_rand < 0.5: # Catmull-Rom
            b = np.random.normal(0, 1/6)
            c = (1 - b) * 0.5
            clip = clip.resize.Bicubic(dw, dh, filter_param_a=b, filter_param_b=c)
        elif abs_rand < 0.65: # standard Bicubic
            b = np.random.normal(1/3, 1/6)
            c = (1 - b) * 0.5
            clip = clip.resize.Bicubic(dw, dh, filter_param_a=b, filter_param_b=c)
        elif abs_rand < 0.8: # sharp Bicubic
            b = np.random.normal(-0.5, 0.25)
            c = b * -0.5
            clip = clip.resize.Bicubic(dw, dh, filter_param_a=b, filter_param_b=c)
        elif abs_rand < 0.9: # soft Bicubic
            b = np.random.normal(0.75, 0.25)
            c = 1 - b
            clip = clip.resize.Bicubic(dw, dh, filter_param_a=b, filter_param_b=c)
        else: # arbitrary Bicubic
            b = np.random.normal(0, 0.5)
            c = np.random.normal(0.25, 0.25)
            clip = clip.resize.Bicubic(dw, dh, filter_param_a=b, filter_param_b=c)
        # random linear-to-gamma
        if rand_val < 0: clip = clip.resize.Bicubic(transfer_s='709', transfer_in_s='linear')
        # return
        return clip
    _dst = _dst_blk.std.FrameEval(eval_func)
    
    def parse2_pyfunc(label):
        channel_index = 0 if data_format == 'NCHW' else -1
        # safely acquire and increase shared index
        _lock.acquire()
        index = _index_ref[0]
        _index_ref[0] = (index + 1) % epoch_size
        _lock.release()
        # processing using vs
        _src_ref[index] = label
        f_dst = _dst.get_frame(index)
        _src_ref[index] = None
        # vs.VideoFrame to np.ndarray
        data = []
        planes = f_dst.format.num_planes
        for p in range(planes):
            f_arr = np.array(f_dst.get_read_array(p), copy=False)
            data.append(f_arr)
        data = np.stack(data, axis=channel_index)
        # noise spatial correlation
        def noise_correlation(noise, corr):
            if corr > 0:
                sigma = np.random.normal(corr, corr)
                if sigma > 0.25:
                    sigma = [0, sigma, sigma] if data_format == 'NCHW' else [sigma, sigma, 0]
                    noise = ndimage.gaussian_filter(noise, sigma, truncate=2.0)
            return noise
        # add Gaussian noise of random scale and random spatial correlation
        if FLAGS.noise_scale > 0:
            rand_val = np.random.uniform(0, 1)
            scale = np.random.exponential(FLAGS.noise_scale)
            if rand_val >= 0.2 and scale > 0.002: # add noise
                noise_shape = list(data.shape)
                if rand_val < 0.35: # RGB noise
                    noise = np.random.normal(0.0, scale, noise_shape).astype(np.float32)
                    noise = noise_correlation(noise, FLAGS.noise_corr)
                else: # Y/YUV noise
                    noise_shape[channel_index] = 1
                    noise_y = np.random.normal(0.0, scale, noise_shape).astype(np.float32)
                    noise_y = noise_correlation(noise_y, FLAGS.noise_corr)
                    scale_uv = np.random.exponential(FLAGS.noise_scale / 2)
                    if rand_val < 0.55 and scale_uv > 0.002: # YUV noise
                        noise_u = np.random.normal(0.0, scale_uv, noise_shape).astype(np.float32)
                        noise_u = noise_correlation(noise_u, FLAGS.noise_corr * 1.5)
                        noise_v = np.random.normal(0.0, scale_uv, noise_shape).astype(np.float32)
                        noise_v = noise_correlation(noise_v, FLAGS.noise_corr * 1.5)
                        rand_val2 = np.random.uniform(0, 1)
                        if rand_val2 < 0.3: # Rec.601
                            Kr = 0.299
                            Kg = 0.587
                            Kb = 0.114
                        elif rand_val2 < 0.9: # Rec.709
                            Kr = 0.2126
                            Kg = 0.7152
                            Kb = 0.0722
                        else: # Rec.2020
                            Kr = 0.2627
                            Kg = 0.6780
                            Kb = 0.0593
                        noise_r = noise_y + ((1 - Kr) / 2) * noise_v
                        noise_b = noise_y + ((1 - Kb) / 2) * noise_u
                        noise_g = (1 / Kg) * noise_y - (Kr / Kg) * noise_r - (Kb / Kg) * noise_b
                        noise = [noise_r, noise_g, noise_b]
                    else:
                        noise = [noise_y, noise_y, noise_y]
                    noise = np.concatenate(noise, axis=channel_index)
                # adding noise
                data += noise
        # return
        return data, label
    
    def parse3_func(data, label):
        # final process
        data = tf.clip_by_value(data, 0.0, 1.0)
        label = tf.clip_by_value(label, 0.0, 1.0)
        # data format conversion
        if data_format == 'NCHW':
            data.set_shape([channels, None, None])
            label.set_shape([channels, None, None])
            #data.set_shape([channels, patch_height // scaling, patch_width // scaling])
            #label.set_shape([channels, patch_height, patch_width])
        else:
            data.set_shape([None, None, channels])
            label.set_shape([None, None, channels])
            #data.set_shape([patch_height // scaling, patch_width // scaling, channels])
            #label.set_shape([patch_height, patch_width, channels])
        # return
        return data, label
    
    # Dataset API
    dataset = tf.contrib.data.Dataset.from_tensor_slices(tf.constant(files))
    dataset = dataset.map(parse1_func, num_threads=threads,
                          output_buffer_size=threads * 64)
    dataset = dataset.map(lambda label: tf.py_func(parse2_pyfunc,
                              [label], [tf.float32, tf.float32]),
                          num_threads=threads_py, output_buffer_size=threads_py * 64)
    dataset = dataset.map(parse3_func, num_threads=threads,
                          output_buffer_size=threads * 64)
    if is_training: dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size)
    if is_training: dataset = dataset.repeat(num_epochs)
    
    # return iterator
    iterator = dataset.make_one_shot_iterator()
    next_data, next_label = iterator.get_next()
    
    return next_data, next_label
