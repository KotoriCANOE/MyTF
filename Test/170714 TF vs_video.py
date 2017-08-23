import numpy as np
import gc

import vapoursynth as vs
import tensorflow as tf

# make directory
def mkdir(path):
    import os
    # remove front blanks
    path = path.strip()
    # remove / and \ at the end
    path = path.rstrip('/')
    path = path.rstrip('\\')

    if os.path.exists(path):
        return False
    else:
        os.makedirs(path)
        return True

# convert uint8 np.ndarray to float32 np.ndarray
def uint8_float32(u8, s=None, mul=None):
    assert isinstance(u8, np.ndarray)

    if s is None:
        s = np.empty(u8.shape, np.float32)
    if mul is None:
        mul = 1 / 255
    np.copyto(s, u8)
    s *= mul
    return s

# convert uint8 np.ndarray to float32 vs.VideoNode
def uint8_vsclip(u8, clip=None, mul=None):
    assert isinstance(u8, np.ndarray)

    core = vs.get_core()
    shape_len = len(u8.shape)
    num = u8.shape[-4] if shape_len >= 4 else 1
    height = u8.shape[-3]
    width = u8.shape[-2]
    planes = u8.shape[-1]
    if clip is None:
        clip = core.std.BlankClip(None, width, height, vs.RGBS if planes == 3 else vs.GRAYS, num)

    def convert_func(n, f):
        fout = f.copy()
        for p in range(planes):
            d = np.array(fout.get_write_array(p), copy=False)
            uint8_float32(u8[n, :, :, p], d, mul)
            del d
        return fout
    return core.std.ModifyFrame(clip, clip, convert_func)

# convert float32 np.ndarray to float32 vs.VideoNode
def float32_vsclip(s, clip=None):
    assert isinstance(s, np.ndarray)

    core = vs.get_core()
    shape_len = len(s.shape)
    num = s.shape[-4] if shape_len >= 4 else 1
    height = s.shape[-3]
    width = s.shape[-2]
    planes = s.shape[-1]
    if clip is None:
        clip = core.std.BlankClip(None, width, height, vs.RGBS if planes == 3 else vs.GRAYS, num)

    def convert_func(n, f):
        fout = f.copy()
        for p in range(planes):
            d = np.array(fout.get_write_array(p), copy=False)
            np.copyto(d, s[n, :, :, p])
            del d
        return fout
    return core.std.ModifyFrame(clip, clip, convert_func)

# convert vs.VideoFrame to np.ndarray
def vsframe_ndarray(frame):
    assert isinstance(frame, vs.VideoFrame)

    planes = frame.format.num_planes
    d = []
    for p in range(planes):
        arr = np.array(frame.get_read_array(p), copy=False)
        d.append(arr)
    d = np.stack(d, axis=2)
    
    return d

# convert vs.VideoNode to np.ndarray
def vsclip_ndarray(clip):
    assert isinstance(clip, vs.VideoNode)

    num = clip.num_frames
    planes = clip.format.num_planes
    dn = []
    for n in range(num):
        f = clip.get_frame(n)
        d = []
        for p in range(planes):
            arr = np.array(f.get_read_array(p), copy=False)
            d.append(arr)
        dn.append(np.stack(d, axis=2))
    dn = np.stack(dn, axis=0)
    
    return dn

# resample clip using zimg resizer
def resample(clip, dw, dh, linear_scale=False, down=6, upfilter=None, noring=False):
    assert isinstance(clip, vs.VideoNode)

    sw = clip.width
    sh = clip.height

    # gamma to linear
    if linear_scale:
        src = clip = clip.resize.Bicubic(transfer_s='linear', transfer_in_s='709')
    
    # down-sampling
    if down == 0:
        clip = clip.resize.Point(dw, dh)
    elif down == 1:
        clip = clip.resize.Bilinear(dw, dh)
    elif down == 2:
        clip = clip.resize.Spline16(dw, dh)
    elif down == 3:
        clip = clip.resize.Spline36(dw, dh)
    elif down == 4:
        clip = clip.resize.Lanczos(dw, dh, filter_param_a=3)
    elif down == 5:
        clip = clip.resize.Bicubic(dw, dh, filter_param_a=-0.5, filter_param_b=0.25)
    elif down == 6:
        clip = clip.resize.Bicubic(dw, dh, filter_param_a=0, filter_param_b=0.5) # Catmull-Rom
    elif down == 7:
        clip = clip.resize.Bicubic(dw, dh, filter_param_a=1/3, filter_param_b=1/3) # Mitchell-Netravali
    elif down == 8:
        clip = clip.resize.Bicubic(dw, dh, filter_param_a=0.3782, filter_param_b=0.3109) # Robidoux
    elif down == 9:
        clip = clip.resize.Bicubic(dw, dh, filter_param_a=1, filter_param_b=0) # SoftCubic100
    else:
        raise ValueError('unknown \'down\'')
        
    # ringing removal
    if noring:
        clip = clip.rgvs.Repair(src.fmtc.resample(dw, dh, kernel='gauss', a1=100), 1)
    
    # linear to gamma
    if linear_scale:
        clip = clip.resize.Bicubic(transfer_s='709', transfer_in_s='linear')

    # up-sampling
    down = clip = clip.std.Limiter()
    if not upfilter:
        pass
    elif upfilter == 'bicubic':
        up = clip.resize.Bicubic(sw, sh, filter_param_a=0, filter_param_b=0.5)
    elif upfilter == 'point':
        up = clip.resize.Point(sw, sh)
    else:
        raise ValueError('unknown \'upfilter\'')
    
    # return
    if upfilter:
        return down, up
    else:
        return down

# reading images using FIFOQueue within tensorflow graph
def ImageReader(files, channels=0, shuffle=False):
    file_queue = tf.train.string_input_producer(files, shuffle=shuffle)
    reader = tf.WholeFileReader()
    key, value = reader.read(file_queue)
    image = tf.image.decode_image(value, channels=channels)
    image.set_shape([None, None, 3])
    return image

# writing batch of images within tensorflow graph
def ImageBatchWriter(sess, images, files, dtype=tf.uint8):
    pngs = []
    for i in range(len(files)):
        img = images[i]
        img = tf.image.convert_image_dtype(img, dtype, saturate=True)
        png = tf.image.encode_png(img, compression=9)
        pngs.append(png)
    pngs = sess.run(pngs)
    for i in range(len(files)):
        with open(files[i], 'wb') as f:
            f.write(pngs[i])

# dataset
filename = r'D:\Record\20170707.mkv'

core = vs.get_core(threads=1)
core.max_cache_size = 2000
clip = core.ffms2.Source(filename, threads=1)
clip = clip.resize.Bicubic(format=vs.RGBS, matrix_in_s='709',
                           filter_param_a=0, filter_param_b=0.5,
                           dither_type='none')
down = resample(clip, clip.width // 2, clip.height // 2, linear_scale=True, down=6)

# dataset mapping function
def read_func(number):
    label = vsframe_ndarray(clip.get_frame(number))
    data = vsframe_ndarray(down.get_frame(number))
    return data, label

# parameters
save_path = './170714'
num_epochs = 10
batch_size = 16
buffer_size = 128

mkdir(save_path)

# Dataset API
dataset = tf.contrib.data.Dataset.range(1000)
dataset = dataset.map(lambda number: tuple(tf.py_func(read_func, [number], [tf.float32, tf.float32])))
dataset = dataset.shuffle(buffer_size=buffer_size)
dataset = dataset.batch(batch_size)
dataset = dataset.repeat(num_epochs)
print(dataset.output_types, dataset.output_shapes)

iterator = dataset.make_one_shot_iterator()
next_data, next_label = iterator.get_next()

with tf.Session() as sess:
    out_files = [save_path + '/{:0>4}.png'.format(i) for i in range(batch_size)]
    data = sess.run(next_data)
    print(data.shape)
    ImageBatchWriter(sess, next_data, out_files)

gc.collect()
