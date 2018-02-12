import os
import numpy as np
import tensorflow as tf

from model import SRmodel

# working directory
print('Current working directory:\n    {}\n'.format(os.getcwd()))

# flags
FLAGS = tf.app.flags.FLAGS

# parameters
tf.app.flags.DEFINE_string('postfix', '',
                            """Postfix added to train_dir, test_dir, test files, etc.""")
tf.app.flags.DEFINE_string('train_dir', './train{postfix}.tmp',
                           """Directory where to read checkpoint.""")
tf.app.flags.DEFINE_string('graph_dir', './model{postfix}.tmp',
                           """Directory where to write meta graph and data.""")

def session():
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options,
        allow_soft_placement=True, log_device_placement=False)
    return tf.Session(config=config)

# build and save graph
def graph():
    # build model
    model = SRmodel(FLAGS, data_format=FLAGS.data_format,
        input_range=FLAGS.input_range, output_range=FLAGS.output_range,
        multiGPU=FLAGS.multiGPU, use_fp16=FLAGS.use_fp16,
        scaling=FLAGS.scaling, image_channels=FLAGS.image_channels)
    
    model.build_model()
    
    with session() as sess:
        # save the GraphDef
        tf.train.write_graph(sess.graph_def, FLAGS.graph_dir, 'model.graphdef',
            as_text=True)
        
        # a Saver object to restore the variables with mappings
        saver = tf.train.Saver(var_list=model.g_rvars)
        
        # restore variables from checkpoint
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.train_dir))
        
        # a Saver object to save the variables without mappings
        saver = tf.train.Saver(var_list=model.g_svars)
        
        # save the model parameters
        saver.export_meta_graph(os.path.join(FLAGS.graph_dir, 'model.meta'),
            as_text=False, clear_devices=True, clear_extraneous_savers=True)
        saver.save(sess, os.path.join(FLAGS.graph_dir, 'model'),
                   write_meta_graph=False, write_state=False)

# stderr print
def eprint(*args, **kwargs):
    import sys
    print(*args, file=sys.stderr, **kwargs)

# main
def main(argv=None):
    import shutil
    # arXiv 1509.09308
    # a new class of fast algorithms for convolutional neural networks using Winograd's minimal filtering algorithms
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    
    FLAGS.train_dir = FLAGS.train_dir.format(postfix=FLAGS.postfix)
    FLAGS.graph_dir = FLAGS.graph_dir.format(postfix=FLAGS.postfix)
    
    if os.path.exists(FLAGS.graph_dir):
        eprint('Removed :' + FLAGS.graph_dir)
        shutil.rmtree(FLAGS.graph_dir)
    os.makedirs(FLAGS.graph_dir)
    with tf.Graph().as_default():
        graph()

if __name__ == '__main__':
    tf.app.run()
