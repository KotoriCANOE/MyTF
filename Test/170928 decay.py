import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# parameters
total_steps = 500000
learning_rate = 1e-3
lr_min = 5e-6
lr_decay_steps = 250
lr_decay_factor = 0.01
global_step = tf.train.get_or_create_global_step()

# training ops
g_train_ops = []

# process
'''
lr_decay_rate = tf.train.exponential_decay(lr_decay_factor, global_step,
    lr_decay_steps, 0.9992, staircase=True)
lr_decay_rate = tf.maximum(lr_decay_factor * 0.5, lr_decay_rate)
lr_decay_rate = 1 - lr_decay_rate
g_lr = tf.train.exponential_decay(learning_rate, global_step,
    lr_decay_steps, lr_decay_rate, staircase=True)
'''

lr_decay_rate = tf.Variable(lr_decay_factor, trainable=False)
lr_decay_rate_new = tf.cond(tf.logical_and(tf.greater(global_step, 0),
    tf.equal(global_step % 50000, 0)),
    lambda: lr_decay_rate * 0.6, lambda: lr_decay_rate)
g_train_ops.append(tf.assign(lr_decay_rate, lr_decay_rate_new, use_locking=True))

g_lr = tf.Variable(learning_rate, trainable=False)
g_lr_new = tf.cond(tf.logical_and(tf.greater(global_step, 0),
    tf.equal(global_step % lr_decay_steps, 0)),
    lambda: g_lr * (1 - lr_decay_rate), lambda: g_lr)
g_train_ops.append(tf.assign(g_lr, g_lr_new, use_locking=True))

if lr_min > 0:
    g_lr = tf.maximum(lr_min, g_lr)

g_train_ops.append(tf.assign_add(global_step, lr_decay_steps, use_locking=True))

with tf.control_dependencies(g_train_ops):
    update_op = tf.no_op()

# create session
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_options)
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

# run steps
steps = np.linspace(0, total_steps - 1, total_steps // lr_decay_steps, dtype=np.int32)
lr = np.empty(steps.shape, np.float32)
decay = np.empty(steps.shape, np.float32)

for s in range(len(steps)):
    ret = sess.run((g_lr, lr_decay_rate, update_op))
    lr[s] = ret[0]
    decay[s] = ret[1]

plt.figure()
plt.plot(steps, lr)
plt.yscale('log')
plt.show()
plt.close()

plt.figure()
plt.plot(steps, decay)
plt.yscale('log')
plt.show()
plt.close()
