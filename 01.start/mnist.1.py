import os
import tensorflow as tf

# working directory
print('Current working directory:\n    {}\n'.format(os.getcwd()))

# constants
DTYPE = tf.float32
WIDTH = 28
HEIGHT = 28
INPUT_DIM = WIDTH * HEIGHT
OUTPUT_DIM = 10
LEARN_RATE = 0.05
ITERATIONS = 1000
BATCH_SIZE = 100

# dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# input
x = tf.placeholder(DTYPE, [None, INPUT_DIM])

# weights and biases
W = tf.Variable(tf.zeros([INPUT_DIM, OUTPUT_DIM]))
b = tf.Variable(tf.zeros([OUTPUT_DIM]))

# model and output
y = x @ W + b

# true output
y_ = tf.placeholder(DTYPE, [None, OUTPUT_DIM])

# loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(
        y_ * tf.log(tf.nn.softmax(y)), reduction_indices=[1]))

# optimization
train_step = tf.train.GradientDescentOptimizer(LEARN_RATE).minimize(cross_entropy)

# initialize
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# training
for _ in range(ITERATIONS):
    batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
    sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})

# testing
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
