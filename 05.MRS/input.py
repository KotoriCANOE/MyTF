import sys
import numpy as np
import tensorflow as tf

def inputs(config, files, labels_file, epoch_size=None, is_training=False, is_testing=False):
    # parameters
    threads = config.threads
    num_labels = config.num_labels
    if is_training: num_epochs = config.num_epochs
    data_format = config.data_format
    batch_size = config.batch_size
    if is_training: buffer_size = config.buffer_size

    # labels
    labels = np.load(labels_file, allow_pickle=False)
    if epoch_size: labels = labels[:epoch_size]
    else: epoch_size = len(labels)
    if is_training: buffer_size = min(buffer_size, epoch_size)
    
    # dataset mapping function
    from scipy import ndimage
    
    def parse1_pyfunc(file, label):
        channel_index = 0 if data_format == 'NCHW' else -1
        data_axis = -1 if data_format == 'NCHW' else -2
        # loading data
        data = np.load(file.decode('utf-8'), allow_pickle=False)
        data = np.expand_dims(data, 0) # height
        data = np.expand_dims(data, channel_index) # channels
        # smoothing
        if config.smoothing > 0:
            #sigma = config.smoothing
            #sigma = np.random.exponential(config.smoothing)
            #sigma = np.random.normal(config.smoothing, config.smoothing / 2)
            #sigma = np.random.lognormal(config.smoothing, 1)
            sigma = float('inf')
            while sigma > config.smoothing * 10:
                sigma = np.random.lognormal(1, 1) * config.smoothing
            if sigma > 0:
                data = ndimage.gaussian_filter1d(data, sigma, axis=data_axis, mode='constant')
        # noise spatial correlation
        def noise_correlation(noise, corr):
            if corr > 0:
                #sigma = np.random.normal(corr, corr)
                sigma = np.random.lognormal(corr, 1)
                if sigma > 0:
                    noise = ndimage.gaussian_filter1d(noise, sigma, axis=data_axis, truncate=3.0)
            return noise
        # add Gaussian noise of random scale and random spatial correlation
        if config.noise_scale > 0:
            rand_val = np.random.uniform(0, 1)
            scale = float('inf')
            while scale > config.noise_scale * 3:
                scale = np.random.exponential(config.noise_scale)
            if rand_val >= 0.05 and scale > 0: # add noise
                noise_shape = list(data.shape)
                noise = np.random.normal(0.0, scale, noise_shape).astype(np.float32)
                noise = noise_correlation(noise, config.noise_corr)
                # add multiplicative noise
                data += noise * (data + config.noise_base)
        # clipping values
        data = np.maximum(data, 0)
        # return
        return data, label
    
    # Dataset API
    dataset = tf.contrib.data.Dataset.from_tensor_slices((files, labels))
    if is_training: dataset = dataset.shuffle(buffer_size)
    dataset = dataset.map(lambda file, label: tuple(tf.py_func(parse1_pyfunc,
                          [file, label], [tf.float32, tf.float32])),
                          num_threads=1 if is_testing else threads, output_buffer_size=threads * 64)
    dataset = dataset.batch(batch_size)
    if is_training: dataset = dataset.repeat(num_epochs)
    else: dataset = dataset.repeat()
    
    # return iterator
    iterator = dataset.make_one_shot_iterator()
    next_data, next_label = iterator.get_next()
    if data_format == 'NCHW':
        next_data.set_shape([None, 1, 1, config.seq_size])
    else:
        next_data.set_shape([None, 1, config.seq_size, 1])
    next_label.set_shape([None, num_labels])
    
    return next_data, next_label
