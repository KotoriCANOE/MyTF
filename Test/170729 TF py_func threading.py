import tensorflow as tf
import numpy as np

def inputs():
    threads = 8
    number = threads * 1000
    num_epochs = 2
    scaling = 2
    patch_width = 1024
    patch_height = 1024

    import threading
    import vapoursynth as vs
    
    _lock = threading.Lock()
    _index_ref = [0]
    _src_ref = [None for _ in range(number)]
    core = vs.get_core(threads=threads)
    _src_blk = core.std.BlankClip(None, patch_width, patch_height,
                                 format=vs.RGBS, length=number)
    _dst_blk = core.std.BlankClip(None, patch_width // scaling, patch_height // scaling,
                                 format=vs.RGBS, length=(1 << 31) - 1)
    
    def src_func(n, f):
        f_out = f.copy()
        planes = f_out.format.num_planes
        # output
        f_out.props.__setattr__('data', str(_src_ref[n] * scaling))
        return f_out
    _src = _src_blk.std.ModifyFrame(_src_blk, src_func)
    
    def eval_func(n):
        sw = patch_width
        sh = patch_height
        dw = sw // scaling
        dh = sh // scaling
        clip = _src
        clip = clip.resize.Bicubic(dw, dh)
        clip = clip.resize.Bicubic(sw, sh)
        clip = clip.resize.Bicubic(dw, dh)
        clip = clip.resize.Bicubic(sw, sh)
        clip = clip.resize.Bicubic(dw, dh)
        clip = clip.resize.Bicubic(sw, sh)
        clip = clip.resize.Bicubic(dw, dh)
        # return
        return clip
    _dst = _dst_blk.std.FrameEval(eval_func)
    
    parse_dict = {'lock': _lock, 'index_ref': _index_ref, 'src_ref': _src_ref,
                  'dst': _dst}

    def parse_pyfunc(data):
        # safely acquire and increase shared index
        _lock.acquire()
        index = _index_ref[0]
        _index_ref[0] = (index + 1) % number
        _lock.release()
        # modify data
        _src_ref[index] = data
        frame = _dst.get_frame(index)
        _src_ref[index] = None
        # get data
        data = int(frame.props.data)
        print(index, data)
        return data

    # Dataset API
    dataset = tf.contrib.data.Dataset.range(number)
    dataset = dataset.map(lambda data: tuple(tf.py_func(parse_pyfunc,
                              [data], [tf.int32])),
                          num_threads=threads, output_buffer_size=threads * 64)
    dataset = dataset.repeat(num_epochs)
    
    # iterator
    iterator = dataset.make_one_shot_iterator()
    next_data = iterator.get_next()
    
    # return
    return next_data

# dataset
data = inputs()

# session
config = tf.ConfigProto(log_device_placement=False)
config.gpu_options.allow_growth = True
with tf.train.MonitoredTrainingSession(config=config) as sess:
    while not sess.should_stop():
        sess.run(data)
