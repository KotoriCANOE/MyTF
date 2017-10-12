import tensorflow as tf
import numpy as np

def parse_func(value):
    return -value

dataset = tf.contrib.data.Dataset.range(32)
dataset = dataset.shuffle(16)
dataset = dataset.map(parse_func, num_threads=8, output_buffer_size=32)
dataset = dataset.batch(16)
dataset = dataset.repeat(8)

iterator = dataset.make_one_shot_iterator()
next = iterator.get_next()

gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_options)
sess = tf.Session(config=config)

while True:
    try:
        print(sess.run(next))
    except tf.errors.OutOfRangeError:
        sess.close()
        break
