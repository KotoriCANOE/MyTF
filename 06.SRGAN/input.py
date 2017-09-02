import tensorflow as tf
import numpy as np

def inputs(config, files, is_training=False, is_testing=False):
    # parameters
    channels = config.image_channels
    threads = config.threads
    threads_py = config.threads_py
    scaling = config.scaling
    if is_training: num_epochs = config.num_epochs
    data_format = config.data_format
    patch_height = config.patch_height
    patch_width = config.patch_width
    batch_size = config.batch_size
    if is_training: buffer_size = config.buffer_size
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
        if is_training and config.pre_down:
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
        elif is_testing and config.pre_down:
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
        # random color augmentation
        if is_training and config.color_augmentation > 0:
            block = tf.image.random_saturation(block, 1 - config.color_augmentation, 1 + config.color_augmentation)
            block = tf.image.random_brightness(block, config.color_augmentation)
            block = tf.image.random_contrast(block, 1 - config.color_augmentation, 1 + config.color_augmentation)
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
    
    def SigmoidInverse(clip, thr=0.5, cont=6.5, epsilon=1e-6):
        assert clip.format.sample_type == vs.FLOAT
        x0 = 1 / (1 + np.exp(cont * thr))
        x1 = 1 / (1 + np.exp(cont * (thr - 1)))
        # thr - log(max(1 / max(x * (x1 - x0) + x0, epsilon) - 1, epsilon)) / cont
        expr = '{thr} 1 x {x1_x0} * {x0} + {epsilon} max / 1 - {epsilon} max log {cont_rec} * -'.format(thr=thr, cont_rec=1 / cont, epsilon=epsilon, x0=x0, x1_x0=x1 - x0)
        return clip.std.Expr(expr)
    
    def SigmoidDirect(clip, thr=0.5, cont=6.5):
        assert clip.format.sample_type == vs.FLOAT
        x0 = 1 / (1 + np.exp(cont * thr))
        x1 = 1 / (1 + np.exp(cont * (thr - 1)))
        # (1 / (1 + exp(cont * (thr - x))) - x0) / (x1 - x0)
        expr = '1 1 {cont} {thr} x - * exp + / {x0} - {x1_x0_rec} *'.format(thr=thr, cont=cont, x0=x0, x1_x0_rec=1 / (x1 - x0))
        return clip.std.Expr(expr)
    
    _lock = threading.Lock()
    _index_ref = [0]
    _src_ref = [None for _ in range(epoch_size)]
    core = vs.get_core(threads=1 if is_testing else threads_py)
    _dscales = list(range(1, 5)) if config.pre_down else [1]
    _src_blk = [core.std.BlankClip(None, patch_width * s, patch_height * s,
                                   format=vs.RGBS, length=epoch_size)
                for s in _dscales]
    _dst_blk = core.std.BlankClip(None, patch_width // scaling, patch_height // scaling,
                                 format=vs.RGBS, length=epoch_size)
    
    def src_frame_func(n, f):
        f_out = f.copy()
        planes = f_out.format.num_planes
        # output
        for p in range(planes):
            f_arr = np.array(f_out.get_write_array(p), copy=False)
            np.copyto(f_arr, _src_ref[n][p, :, :] if data_format == 'NCHW' else _src_ref[n][:, :, p])
        return f_out
    _srcs = [s.std.ModifyFrame(s, src_frame_func) for s in _src_blk]
    _srcs_linear = [s.resize.Bicubic(transfer_s='linear', transfer_in_s='709') for s in _srcs]
    
    def src_down_func(clip):
        dw = patch_width
        dh = patch_height
        #clip = clip.resize.Bicubic(transfer_s='linear', transfer_in_s='709')
        clip = SigmoidInverse(clip)
        clip = clip.resize.Bicubic(dw, dh, filter_param_a=0, filter_param_b=0.5)
        clip = SigmoidDirect(clip)
        clip = clip.resize.Bicubic(transfer_s='709', transfer_in_s='linear')
        return clip
    if config.pre_down:
        _srcs_down = [src_down_func(s) for s in _srcs_linear]
    
    def src_select_eval(n):
        # select source
        shape = _src_ref[n].shape
        sh = shape[-2 if data_format == 'NCHW' else -3]
        dscale = sh // patch_height
        # downscale if needed
        if dscale > 1:
            clip = _srcs_down[dscale - 1]
        else:
            clip = _srcs[dscale - 1]
        return clip
    if config.pre_down:
        _src = _src_blk[0].std.FrameEval(src_select_eval)
    else:
        _src = _srcs[0]
    
    def resize_set_func(clip, linear=False):
        # parameters
        dw = patch_width // scaling
        dh = patch_height // scaling
        rets = {}
        # resizers
        rets['bilinear'] = clip.resize.Bilinear(dw, dh)
        rets['spline16'] = clip.resize.Spline16(dw, dh)
        rets['spline36'] = clip.resize.Spline36(dw, dh)
        for taps in range(2, 12):
            rets['lanczos{}'.format(taps)] = clip.resize.Lanczos(dw, dh, filter_param_a=taps)
        # linear to gamma
        if linear:
            for key in rets:
                rets[key] = rets[key].resize.Bicubic(transfer_s='709', transfer_in_s='linear')
        return rets
    _resizes = [resize_set_func(s, linear=False) for s in _srcs]
    _linear_resizes = [resize_set_func(s, linear=True) for s in _srcs_linear]
    
    def resize_eval(n):
        # parameters
        dw = patch_width // scaling
        dh = patch_height // scaling
        rand_val = np.random.uniform(-1, 1)
        abs_rand = np.abs(rand_val)
        # select source
        shape = _src_ref[n].shape
        sh = shape[-2 if data_format == 'NCHW' else -3]
        dscale = sh // patch_height
        # random gamma-to-linear
        if rand_val < 0:
            clip = _srcs_linear[dscale - 1]
            resizes = _linear_resizes[dscale - 1]
        else:
            clip = _srcs[dscale - 1]
            resizes = _resizes[dscale - 1]
        # random resizers
        if abs_rand < 0.05:
            clip = resizes['bilinear']
        elif abs_rand < 0.1:
            clip = resizes['spline16']
        elif abs_rand < 0.15:
            clip = resizes['spline36']
        elif abs_rand < 0.4: # Lanczos taps=[2, 12)
            taps = int(np.clip(np.random.exponential(2) + 2, 2, 11))
            clip = resizes['lanczos{}'.format(taps)]
        elif abs_rand < 0.6: # Catmull-Rom
            b = np.random.normal(0, 1/6)
            c = (1 - b) * 0.5
            clip = clip.resize.Bicubic(dw, dh, filter_param_a=b, filter_param_b=c)
        elif abs_rand < 0.7: # Mitchell-Netravali (standard Bicubic)
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
        if rand_val < 0 and abs_rand >= 0.4:
            clip = clip.resize.Bicubic(transfer_s='709', transfer_in_s='linear')
        # return
        return clip
    _dst = _dst_blk.std.FrameEval(resize_eval)
    
    def parse2_pyfunc(label):
        channel_index = -3 if data_format == 'NCHW' else -1
        dscale = label.shape[-2 if data_format == 'NCHW' else -3] // patch_height
        # safely acquire and increase shared index
        _lock.acquire()
        index = _index_ref[0]
        _index_ref[0] = (index + 1) % epoch_size
        _lock.release()
        # processing using vs
        _src_ref[index] = label
        if config.pre_down and dscale > 1: f_src = _src.get_frame(index)
        f_dst = _dst.get_frame(index)
        _src_ref[index] = None
        # vs.VideoFrame to np.ndarray
        if config.pre_down and dscale > 1:
            label = []
            planes = f_src.format.num_planes
            for p in range(planes):
                f_arr = np.array(f_src.get_read_array(p), copy=False)
                label.append(f_arr)
            label = np.stack(label, axis=channel_index)
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
        if config.noise_scale > 0:
            rand_val = np.random.uniform(0, 1)
            scale = np.random.exponential(config.noise_scale)
            if rand_val >= 0.2 and scale > 0.002: # add noise
                noise_shape = list(data.shape)
                if rand_val < 0.35: # RGB noise
                    noise = np.random.normal(0.0, scale, noise_shape).astype(np.float32)
                    noise = noise_correlation(noise, config.noise_corr)
                else: # Y/YUV noise
                    noise_shape[channel_index] = 1
                    noise_y = np.random.normal(0.0, scale, noise_shape).astype(np.float32)
                    noise_y = noise_correlation(noise_y, config.noise_corr)
                    scale_uv = np.random.exponential(config.noise_scale / 2)
                    if rand_val < 0.55 and scale_uv > 0.002: # YUV noise
                        noise_u = np.random.normal(0.0, scale_uv, noise_shape).astype(np.float32)
                        noise_u = noise_correlation(noise_u, config.noise_corr * 1.5)
                        noise_v = np.random.normal(0.0, scale_uv, noise_shape).astype(np.float32)
                        noise_v = noise_correlation(noise_v, config.noise_corr * 1.5)
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
        # JPEG encoding
        if config.jpeg_coding:
            rand_val = tf.random_uniform([], 0, 1, seed=config.random_seed if is_testing else None)
            def c_f1(data):
                if data_format == 'NCHW':
                    data = tf.transpose(data, (1, 2, 0))
                data = tf.image.convert_image_dtype(data, tf.uint8, saturate=True)
                def _f1(quality):
                    return tf.image.encode_jpeg(data, quality=quality, chroma_downsampling=False)
                data = tf.cond(tf.less(rand_val, 0.02), lambda: _f1(100),
                    lambda: tf.cond(tf.less(rand_val, 0.04), lambda: _f1(99),
                    lambda: tf.cond(tf.less(rand_val, 0.06), lambda: _f1(98),
                    lambda: tf.cond(tf.less(rand_val, 0.08), lambda: _f1(97),
                    lambda: tf.cond(tf.less(rand_val, 0.10), lambda: _f1(96),
                    lambda: tf.cond(tf.less(rand_val, 0.12), lambda: _f1(95),
                    lambda: tf.cond(tf.less(rand_val, 0.14), lambda: _f1(94),
                    lambda: tf.cond(tf.less(rand_val, 0.16), lambda: _f1(93),
                    lambda: tf.cond(tf.less(rand_val, 0.18), lambda: _f1(92),
                    lambda: tf.cond(tf.less(rand_val, 0.20), lambda: _f1(91),
                    lambda: tf.cond(tf.less(rand_val, 0.22), lambda: _f1(90),
                    lambda: tf.cond(tf.less(rand_val, 0.24), lambda: _f1(89),
                    lambda: tf.cond(tf.less(rand_val, 0.26), lambda: _f1(88),
                    lambda: tf.cond(tf.less(rand_val, 0.28), lambda: _f1(87),
                    lambda: tf.cond(tf.less(rand_val, 0.30), lambda: _f1(86),
                                                             lambda: _f1(85)
                    )))))))))))))))
                data = tf.image.decode_jpeg(data)
                data = tf.image.convert_image_dtype(data, tf.float32, saturate=False)
                if data_format == 'NCHW':
                    data = tf.transpose(data, (2, 0, 1))
                return data
            data = tf.cond(tf.less(rand_val, 0.32), lambda: c_f1(data), lambda: data)
        # return
        return data, label
    
    # Dataset API
    dataset = tf.contrib.data.Dataset.from_tensor_slices((files))
    if is_training and buffer_size > 0: dataset = dataset.shuffle(buffer_size)
    dataset = dataset.map(parse1_func, num_threads=threads,
                          output_buffer_size=threads * 64)
    dataset = dataset.map(lambda label: tuple(tf.py_func(parse2_pyfunc,
                              [label], [tf.float32, tf.float32])),
                          num_threads=1 if is_testing else threads_py, output_buffer_size=threads_py * 64)
    dataset = dataset.map(parse3_func, num_threads=1 if is_testing else threads,
                          output_buffer_size=threads * 64)
    dataset = dataset.batch(batch_size)
    if is_training and num_epochs > 0: dataset = dataset.repeat(num_epochs)
    
    # return iterator
    iterator = dataset.make_one_shot_iterator()
    next_data, next_label = iterator.get_next()
    
    # data shape declaration
    if data_format == 'NCHW':
        next_data.set_shape([None, channels, None, None])
        next_label.set_shape([None, channels, None, None])
        #next_data.set_shape([None, channels, patch_height // scaling, patch_width // scaling])
        #next_label.set_shape([None, channels, patch_height, patch_width])
    else:
        next_data.set_shape([None, None, None, channels])
        next_label.set_shape([None, None, None, channels])
        #next_data.set_shape([None, patch_height // scaling, patch_width // scaling, channels])
        #next_label.set_shape([None, patch_height, patch_width, channels])
    
    return next_data, next_label
