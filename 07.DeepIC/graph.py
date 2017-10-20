import os
import numpy as np
import tensorflow as tf

from model import ICmodel

# working directory
print('Current working directory:\n    {}\n'.format(os.getcwd()))

# flags
FLAGS = tf.app.flags.FLAGS

# parameters
tf.app.flags.DEFINE_string('postfix', '',
                            """Postfix added to train_dir, test_dir, test files, etc.""")
tf.app.flags.DEFINE_string('train_dir', './train{}.tmp'.format(FLAGS.postfix),
                           """Directory where to read checkpoint.""")
tf.app.flags.DEFINE_string('graph_dir', './graph.tmp',
                           """Directory where to write meta graph and data.""")

# build and save graph
def graph():
    with tf.Graph().as_default():
        # build model
        model = ICmodel(FLAGS, data_format=FLAGS.data_format,
            input_range=FLAGS.input_range, output_range=FLAGS.output_range,
            qp_range=FLAGS.qp_range, multiGPU=FLAGS.multiGPU, use_fp16=FLAGS.use_fp16,
            image_channels=FLAGS.image_channels)
        
        model.build_model()
        
        # a saver object which will save all the variables
        saver = tf.train.Saver(var_list=model.g_svars)
        
        # create session
        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(gpu_options=gpu_options,
            log_device_placement=False)
        sess = tf.Session(config=config)
        
        # restore variables from checkpoint
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.train_dir))
        
        # save the graph
        saver.export_meta_graph(os.path.join(FLAGS.graph_dir, 'model.meta'),
            as_text=False, clear_devices=True, clear_extraneous_savers=True)
        saver.save(sess, os.path.join(FLAGS.graph_dir, 'model'),
                   write_meta_graph=False, write_state=False)

# main
def main(argv=None):
    # arXiv 1509.09308
    # a new class of fast algorithms for convolutional neural networks using Winograd's minimal filtering algorithms
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    
    if tf.gfile.Exists(FLAGS.graph_dir):
        tf.gfile.DeleteRecursively(FLAGS.graph_dir)
    tf.gfile.MakeDirs(FLAGS.graph_dir)
    with tf.Graph().as_default():
        graph()

if __name__ == '__main__':
    tf.app.run()
