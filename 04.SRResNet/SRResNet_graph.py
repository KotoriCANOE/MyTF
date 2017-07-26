import os
import numpy as np
import tensorflow as tf

import SRResNet as model

# working directory
print('Current working directory:\n    {}\n'.format(os.getcwd()))

# flags
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('postfix', '',
                            """Postfix added to train_dir, test_dir, test files, etc.""")
tf.app.flags.DEFINE_string('train_dir', './train{}.tmp'.format(FLAGS.postfix),
                           """Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_string('graph_dir', './graph.tmp',
                           """Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_integer('threads', 8,
                            """Number of threads for Dataset process.""")
tf.app.flags.DEFINE_string('data_format', 'NCHW', # 'NHWC'
                            """Data layout format.""")

# build and save graph
def graph():
    # place holder
    input = tf.placeholder(tf.float32, (None, FLAGS.image_channels, None, None), name='Input')
    
    # model inference
    output = model.inference(input, is_training=False)
    
    # a saver object which will save all the variables
    saver = tf.train.Saver()
    
    # create session
    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    
    # restore variables from checkpoint
    saver.restore(sess, tf.train.latest_checkpoint(FLAGS.train_dir))
    
    # save the graph
    saver.export_meta_graph(os.path.join(FLAGS.graph_dir, 'model.pbtxt'), as_text=True)
    saver.save(sess, os.path.join(FLAGS.graph_dir, 'model'),
               write_meta_graph=True, write_state=False)

# main
def main(argv=None):
    if tf.gfile.Exists(FLAGS.graph_dir):
        tf.gfile.DeleteRecursively(FLAGS.graph_dir)
    tf.gfile.MakeDirs(FLAGS.graph_dir)
    with tf.Graph().as_default():
        graph()

if __name__ == '__main__':
    tf.app.run()
