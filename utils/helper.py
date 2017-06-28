import tensorflow as tf

# functions
def divUp(dividend, divisor):
    return (dividend + divisor - 1) // divisor

LayerNum = 0

def conv2dLayer(x, out_channels=None, kernel=3, stride=1, padding='SAME', pooling=1, activation='relu',
                weight_std=1e-3, bias=0, wd=None, name=None):
    global LayerNum
    LayerNum += 1
    if name is None: name = str(LayerNum)
    
    in_channels = x.shape[-1]
    if out_channels is None: out_channels = in_channels
    W = tf.Variable(tf.truncated_normal([kernel, kernel, in_channels, out_channels],
                                        stddev=weight_std, dtype=x.dtype), name='weight' + name)
    b = tf.Variable(tf.constant(bias, x.dtype, [out_channels]), name='bias' + name)
    last = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding, name='conv' + name)
    last = tf.nn.bias_add(last, b, name='conv_bias' + name)
    
    if pooling > 1:
        last = tf.nn.max_pool(last, ksize=[1, pooling, pooling, 1], strides=[1, pooling, pooling, 1],
                              padding=padding, name='pool' + name)
    if activation == 'relu':
        last = tf.nn.relu(last, name='relu' + name)
    elif activation:
        raise ValueError('Unsupported \'activation\' specified!')
    
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(W), wd, name='weight_loss' + name)
        tf.add_to_collection('losses', weight_decay)
    
    return last, W, b

def deconv2dLayer(x, out_channels=None, kernel=3, stride=2, padding='SAME', activation='',
                  weight_std=1e-3, bias=0, wd=None, name=None):
    global LayerNum
    LayerNum += 1
    if name is None: name = str(LayerNum)
    
    in_channels = x.shape[-1]
    if out_channels is None: out_channels = in_channels
    out_shape = (x.shape[0], x.shape[1] * stride, x.shape[2] * stride, out_channels)
    W = tf.Variable(tf.truncated_normal([kernel, kernel, out_channels, in_channels],
                                        stddev=weight_std, dtype=x.dtype))
    b = tf.Variable(tf.constant(bias, x.dtype, [out_channels]))
    last = tf.nn.conv2d_transpose(x, W, output_shape=out_shape, strides=[1, stride, stride, 1],
                                  padding=padding, name='deconv' + name)
    last = tf.nn.bias_add(last, b, name='conv_bias' + name)
    
    if activation == 'relu':
        last = tf.nn.relu(last, name='relu' + name)
    elif activation:
        raise ValueError('Unsupported \'activation\' specified!')
    
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(W), wd, name='weight_loss' + name)
        tf.add_to_collection('losses', weight_decay)
    
    return last, W, b
