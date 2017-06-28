import numpy as np
import h5py
import tensorflow as tf

class SRData(object):
    pass

def read(filename, dtype=None):
    def readDataset(d, dtype=None):
        dataset = np.empty(d.shape, d.dtype)
        d.read_direct(dataset)
        if dtype is not None and dataset.dtype.name != dtype.name:
            print('Data are casted to ' + dtype.name)
            dataset = tf.cast(dataset, dtype)
        return dataset
    dataTag = 'data'
    data2Tag = 'data2'
    labelTag = 'label'
    dataset = SRData()
    with h5py.File(filename, 'r') as file:
        dataset.data = readDataset(file[dataTag], dtype)
        dataset.data2 = readDataset(file[data2Tag], dtype)
        dataset.label = readDataset(file[labelTag], dtype)
        shape = dataset.data.shape
        print('dataset.data.shape={}'.format(shape))
        dataset.number = shape[0]
        dataset.height = shape[1]
        dataset.width = shape[2]
        dataset.channels = shape[3]
    return dataset

def generate_batch(dataset, batch_size, min_queue_examples,
                   enqueue_many, shuffle=False, threads=1):
    datasets = [dataset.data, dataset.data2, dataset.label]
    if shuffle:
        batches = tf.train.shuffle_batch(
                datasets, batch_size,
                capacity=min_queue_examples + 3 * batch_size,
                min_after_dequeue=min_queue_examples,
                num_threads=threads, enqueue_many=enqueue_many)
    else:
        batches = tf.train.batch(
                datasets, batch_size,
                capacity=min_queue_examples + 3 * batch_size,
                num_threads=threads, enqueue_many=enqueue_many)
    #tf.summary.image('data', batches[0])
    #tf.summary.image('data2', batches[1])
    #tf.summary.image('label', batches[2])
    return batches

def inputs(filename, batch_size, shuffle=False, dtype=None, threads=1):
    dataset = read(filename, dtype)
    # ensure that the random shuffling has good mixing properties
    epoch_size = dataset.number
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(epoch_size * min_fraction_of_examples_in_queue)
    # generate a batch of images and labels by building up a queue of examples
    batches = generate_batch(dataset, batch_size, min_queue_examples,
                   enqueue_many=True, shuffle=shuffle, threads=threads)
    return tuple(list(batches) + [epoch_size])
