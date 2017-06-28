import os
from datetime import datetime
import tensorflow as tf

# functions
def divUp(dividend, divisor):
    return (dividend + divisor - 1) // divisor

def weightVariable(shape, dtype=tf.float32):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1, dtype=dtype))

def biasVariable(shape, dtype=tf.float32):
    return tf.Variable(tf.constant(0.1, dtype, shape))

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def maxPool2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def calculateMemoryCNN(dtype, channels, conv_num, pool_layers, width, height, size):
    input_dim = width * height * size
    memory = channels[0] * input_dim
    for i in range(conv_num):
        memory += channels[i + 1] * input_dim * 2 # x2 for activation
        if pool_layers[i]:
            width = divUp(width, 2)
            height = divUp(height, 2)
            input_dim = width * height * size
            memory += channels[i + 1] * input_dim
    memory *= dtype.size
    memory = divUp(memory, 1 << 20) # Mega Bytes
    return memory

# working directory
print('Current working directory:\n    {}\n'.format(os.getcwd()))

# constants
MEMORY_LIMIT = 1024
DTYPE = tf.float32
WIDTH = 28
HEIGHT = 28
INPUT_DIM = WIDTH * HEIGHT
OUTPUT_DIM = 10

# model
def fit_model(CONV_NUM, POOL_LAYERS, CHANNELS, LEARN_RATE, EPOCHS, BATCH_SIZE):
    # constants
    POOL_NUM = sum(POOL_LAYERS)
    LAYERS_NUM = len(CHANNELS) - 1
    
    # dataset
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
    
    # input and true output
    x = tf.placeholder(DTYPE, [None, INPUT_DIM])
    y_ = tf.placeholder(DTYPE, [None, OUTPUT_DIM])
    
    # pre-process
    x_image = tf.reshape(x, [-1, HEIGHT, WIDTH, CHANNELS[0]])
    
    # parameters
    Params = [{} for i in range(LAYERS_NUM)]
    Layers = [{} for i in range(LAYERS_NUM)]
    
    # convolutional layers with activation and pooling
    def conv_layers(input, begin, number, activation=None):
        if activation is None: activation = 1
        if type(activation) in (bool, int):
            activation = [0 for i in range(begin)] + [activation for i in range(number)]
        for i in range(begin, number):
            Params[i]['W'] = weightVariable([3, 3, CHANNELS[i], CHANNELS[i + 1]])
            Params[i]['b'] = biasVariable([CHANNELS[i + 1]])
            last = Layers[i - 1]['output'] if i > begin else input
            last = Layers[i]['conv'] = conv2d(last, Params[i]['W']) + Params[i]['b']
            if POOL_LAYERS[i]:
                last = Layers[i]['pool'] = maxPool2x2(last)
            if activation[i]:
                last = Layers[i]['activate'] = tf.nn.relu(last)
            Layers[i]['output'] = last
            return last
    last = conv_layers(x_image, 0, CONV_NUM)
    
    # densly connected layer
    input_dim = divUp(HEIGHT, 1 << POOL_NUM) * divUp(WIDTH, 1 << POOL_NUM) * CHANNELS[CONV_NUM]
    Params[CONV_NUM]['W'] = weightVariable([input_dim, CHANNELS[CONV_NUM + 1]])
    Params[CONV_NUM]['b'] = biasVariable([CHANNELS[CONV_NUM + 1]])
    last = tf.reshape(last, [-1, input_dim])
    last = Layers[CONV_NUM]['fc'] = tf.nn.relu(last @ Params[CONV_NUM]['W'] + Params[CONV_NUM]['b'])
    
    # dropout
    keep_prob = tf.placeholder(DTYPE)
    last = Layers[CONV_NUM]['fc_drop'] = tf.nn.dropout(last, keep_prob)
    
    # readout layer
    Params[CONV_NUM + 1]['W'] = weightVariable([CHANNELS[CONV_NUM + 1], CHANNELS[CONV_NUM + 2]])
    Params[CONV_NUM + 1]['b'] = biasVariable([CHANNELS[CONV_NUM + 2]])
    y = last @ Params[CONV_NUM + 1]['W'] + Params[CONV_NUM + 1]['b']
    
    # loss function
    cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    
    # optimization
    train_step = tf.train.AdamOptimizer(LEARN_RATE).minimize(cross_entropy)
    
    # accuracy
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # training and testing session
    def session(train_data, test_data, batch_size, epochs):
        train_size = train_data.num_examples
        test_size = test_data.num_examples
        iterations = divUp(train_size * epochs, batch_size)
        
        # testing
        test_total_size = calculateMemoryCNN(DTYPE, CHANNELS, CONV_NUM, POOL_LAYERS,
                                             WIDTH, HEIGHT, test_size)
        test_batch_size = divUp(test_size, divUp(test_total_size, MEMORY_LIMIT))
        print('\nTest set size: {}\nTest inference total size: {}MB\nTest batch size: {}\n'
              .format(test_size, test_total_size, test_batch_size))
        def test(epoch=None):
            test_accuracy = []
            for i in range(divUp(test_size, test_batch_size)):
                batch = test_data.next_batch(test_batch_size)
                test_accuracy.append(accuracy.eval(
                        feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0}))
            test_accuracy = sum(test_accuracy) / len(test_accuracy)
            epoch_str = '' if epoch is None else 'epoch {}, '.format(epoch)
            print(epoch_str + 'test accuracy: {}'.format(test_accuracy))
        
        # session
        with tf.Session() as sess:
            # initialize
            sess.run(tf.global_variables_initializer())
            # start
            dt_start = datetime.now()
            print('\nEpochs: {}\nBatch Size: {}\nIterations: {}\n\nStart time: {}\n'
                  .format(epochs, batch_size, iterations, dt_start))
            # training
            for i in range(iterations):
                batch = mnist.train.next_batch(batch_size)
                if i * batch_size % train_size < batch_size:
                    epoch = i * batch_size // train_size
                    #train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
                    #print('epoch {}, training accuracy {}'.format(epoch, train_accuracy))
                    test(epoch)
                train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            # end
            dt_end = datetime.now()
            print('\nEnd time: {}\nDuration: {}\n'.format(dt_end, dt_end - dt_start))
            # testing
            test(epochs)
    
    # run session
    session(mnist.train, mnist.test, batch_size=BATCH_SIZE, epochs=EPOCHS)

# fit model
'''
fit_model(CONV_NUM=2, POOL_LAYERS=[1, 1], CHANNELS=[1, 32, 64, 512, OUTPUT_DIM],
            LEARN_RATE=3e-5, EPOCHS=200, BATCH_SIZE=100)
fit_model(CONV_NUM=2, POOL_LAYERS=[1, 1], CHANNELS=[1, 32, 64, 1024, OUTPUT_DIM],
            LEARN_RATE=3e-5, EPOCHS=200, BATCH_SIZE=100)
fit_model(CONV_NUM=2, POOL_LAYERS=[1, 1], CHANNELS=[1, 32, 64, 2048, OUTPUT_DIM],
            LEARN_RATE=3e-5, EPOCHS=200, BATCH_SIZE=100)
fit_model(CONV_NUM=2, POOL_LAYERS=[1, 1], CHANNELS=[1, 64, 128, 1024, OUTPUT_DIM],
            LEARN_RATE=3e-5, EPOCHS=200, BATCH_SIZE=100)
fit_model(CONV_NUM=3, POOL_LAYERS=[0, 1, 1], CHANNELS=[1, 16, 32, 64, 1024, OUTPUT_DIM],
            LEARN_RATE=3e-5, EPOCHS=200, BATCH_SIZE=100)
fit_model(CONV_NUM=4, POOL_LAYERS=[0, 0, 1, 1], CHANNELS=[1, 16, 16, 32, 64, 1024, OUTPUT_DIM],
            LEARN_RATE=3e-5, EPOCHS=200, BATCH_SIZE=100)
'''
fit_model(CONV_NUM=4, POOL_LAYERS=[0, 0, 1, 1], CHANNELS=[1, 8, 16, 32, 64, 128, OUTPUT_DIM],
            LEARN_RATE=3e-5, EPOCHS=200, BATCH_SIZE=100)
