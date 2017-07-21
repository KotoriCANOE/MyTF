import sys
import numpy as np
import tensorflow as tf

# flags
FLAGS = tf.app.flags.FLAGS

def inputs(files, labels_file, is_training=False):
    # parameters
    threads = FLAGS.threads
    num_labels = FLAGS.num_labels
    if is_training: num_epochs = FLAGS.num_epochs
    batch_size = FLAGS.batch_size
    if is_training: buffer_size = FLAGS.buffer_size

    # labels
    labels = np.load(labels_file, allow_pickle=False)
    
    # dataset mapping function
    def parse1_pyfunc(file, label):
        data = np.load(file.decode('utf-8'), allow_pickle=False)
        data = data.reshape((1, data.shape[0], 1))
        return data, label
    
    def parse2_func(data, label):
        data = tf.expand_dims(data, -1)
        data = tf.expand_dims(data, 1)
        return data, label
    
    # Dataset API
    epoch_size = len(files) - len(files) % batch_size
    files = files[:epoch_size]
    dataset = tf.contrib.data.Dataset.from_tensor_slices(
            (tf.constant(files), labels))
    dataset = dataset.map(lambda file, label: tf.py_func(parse1_pyfunc,
                          [file, label], [tf.float32, tf.float32]),
                          num_threads=threads, output_buffer_size=threads * 64)
    #dataset = dataset.map(parse2_func, num_threads=threads,
    #                      output_buffer_size=threads * 64)
    if is_training: dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size)
    if is_training: dataset = dataset.repeat(num_epochs)

    # return iterator
    iterator = dataset.make_one_shot_iterator()
    next_data, next_label = iterator.get_next()
    next_data.set_shape([batch_size, 1, FLAGS.seq_size, 1])
    next_label.set_shape([batch_size, num_labels])
    
    return next_data, next_label
