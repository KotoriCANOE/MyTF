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

# recursively list all the files' path under directory
def listdir_files(path, recursive=True, filter_ext=None, encoding=None):
    import os, locale
    if encoding is True: encoding = locale.getpreferredencoding()
    if filter_ext is not None: filter_ext = [e.lower() for e in filter_ext]
    files = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        for f in filenames:
            if os.path.splitext(f)[1].lower() in filter_ext:
                file_path = os.path.join(dir_path, f)
                if encoding: file_path = file_path.encode(encoding)
                files.append(file_path)
        if not recursive: break
    return files

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
    if len(u8.shape) <= 3: u8 = u8.reshape([1] + list(u8.shape))
    num = u8.shape[-4]
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
    if len(s.shape) <= 3: s = s.reshape([1] + list(s.shape))
    num = s.shape[-4]
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
        d.append(arr.reshape(list(arr.shape) + [1]))
    d = np.concatenate(d, axis=2)
    
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
            d.append(arr.reshape([1] + list(arr.shape) + [1]))
        dn.append(np.concatenate(d, axis=3))
    dn = np.concatenate(dn, axis=0)
    
    return dn

# resample clip using zimg resizer
def resample(clip, dw, dh, linear_scale=False, down_filter=6, up_filter=None, noring=False):
    assert isinstance(clip, vs.VideoNode)

    sw = clip.width
    sh = clip.height

    # gamma to linear
    if linear_scale:
        src = clip = clip.resize.Bicubic(transfer_s='linear', transfer_in_s='709')
    
    # down-sampling
    if down_filter == 0:
        clip = clip.resize.Point(dw, dh)
    elif down_filter == 1:
        clip = clip.resize.Bilinear(dw, dh)
    elif down_filter == 2:
        clip = clip.resize.Spline16(dw, dh)
    elif down_filter == 3:
        clip = clip.resize.Spline36(dw, dh)
    elif down_filter == 4:
        clip = clip.resize.Lanczos(dw, dh, filter_param_a=3)
    elif down_filter == 5:
        clip = clip.resize.Bicubic(dw, dh, filter_param_a=-0.5, filter_param_b=0.25)
    elif down_filter == 6:
        clip = clip.resize.Bicubic(dw, dh, filter_param_a=0, filter_param_b=0.5) # Catmull-Rom
    elif down_filter == 7:
        clip = clip.resize.Bicubic(dw, dh, filter_param_a=1/3, filter_param_b=1/3) # Mitchell-Netravali
    elif down_filter == 8:
        clip = clip.resize.Bicubic(dw, dh, filter_param_a=0.3782, filter_param_b=0.3109) # Robidoux
    elif down_filter == 9:
        clip = clip.resize.Bicubic(dw, dh, filter_param_a=1, filter_param_b=0) # SoftCubic100
    else:
        raise ValueError('unknown \'down_filter\'')
        
    # ringing removal
    if noring:
        clip = clip.rgvs.Repair(src.fmtc.resample(dw, dh, kernel='gauss', a1=100), 1)
    
    # linear to gamma
    if linear_scale:
        clip = clip.resize.Bicubic(transfer_s='709', transfer_in_s='linear')

    # up-sampling
    if not up_filter:
        return clip
    
    if up_filter == 'bicubic':
        up = clip.resize.Bicubic(sw, sh, filter_param_a=0, filter_param_b=0.5)
    elif up_filter == 'point':
        up = clip.resize.Point(sw, sh)
    else:
        raise ValueError('unknown \'up_filter\'')
    return clip, up

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

# parameters
'''
pre_process = False
files = listdir_files(r'E:\MyTF\Dataset.SR\Train', filter_ext=['.jpeg', '.jpg', '.png'],
                      encoding=None if pre_process else True)
print('number of files: {}'.format(len(files)))
save_path = './170716'
threads = 4
scaling = 2
channels = 3
num_epochs = 1
batch_size = 256
shuffle = True
buffer_size = 4096
block_size = 160
'''
pre_process = True
files = listdir_files(r'E:\MyTF\Dataset.SR\Test', filter_ext=['.jpeg', '.jpg', '.png'],
                      encoding=None if pre_process else True)
print('number of files: {}'.format(len(files)))
save_path = './170716'
threads = 4
scaling = 2
channels = 3
num_epochs = 1
batch_size = 64
shuffle = False
buffer_size = 1024
block_size = 256

# remove invalid files
if pre_process:
    from scipy import ndimage
    import os
    for f in files:
        img = ndimage.imread(f)
        shape = img.shape
        if len(shape) == 0:
            print('Deleting {}\n    shape={}'.format(f, shape))
            try:
                os.remove(f)
            except Exception as err:
                print(err)
        del img
    exit()

# vapoursynth
core = vs.get_core(threads=1)
core.max_cache_size = 2000

# dataset mapping function
def parse1_func(filename):
    tf.Print(filename, [filename])
    dtype = tf.float32
    image = tf.read_file(filename)
    image = tf.image.decode_image(image, channels=channels)
    image.set_shape([None, None, channels])
    shape = image.get_shape()
    height = shape[-3]
    width = shape[-2]
    if (width >= 3072 and height >= 1536) or (width >= 1536 and height >= 3072):
        dscale = 4
    elif (width >= 2048 and height >= 1024) or (width >= 1024 and height >= 2048):
        dscale = 3
    elif (width >= 1024 and height >= 512) or (width >= 512 and height >= 1024):
        dscale = 2
    else:
        dscale = 1
    block = tf.random_crop(image, [block_size * dscale, block_size * dscale, channels])
    block = tf.image.random_flip_up_down(block)
    block = tf.image.random_flip_left_right(block)
    block = tf.image.convert_image_dtype(block, dtype, saturate=False)
    block = tf.image.random_saturation(block, .95, 1.05)
    block = tf.image.random_brightness(block, .05)
    block = tf.image.random_contrast(block, .95, 1.05)
    return block
    
def parse2_pyfunc(block):
    clip = float32_vsclip(block)
    clip = clip.resize.Bicubic(transfer_s='linear', transfer_in_s='709')
    data = resample(clip, block_size // scaling, block_size // scaling, linear_scale=False, down_filter=6)
    data = data.resize.Bicubic(transfer_s='709', transfer_in_s='linear')
    data = vsframe_ndarray(data.get_frame(0))
    if clip.width == block_size and clip.height == block_size:
        label = block
    else:
        label = resample(clip, block_size, block_size, linear_scale=False, down_filter=6)
        label = label.resize.Bicubic(transfer_s='709', transfer_in_s='linear')
        label = vsframe_ndarray(label.get_frame(0))
    del clip
    gc.collect()
    return data, label

def parse3_func(data, label):
    data_shape = [block_size // scaling, block_size // scaling, channels]
    noise = tf.random_normal(data_shape, mean=0.0, stddev=0.005, dtype=data.dtype)
    data = tf.add(data, noise)
    data = tf.clip_by_value(data, 0.0, 1.0)
    label = tf.clip_by_value(label, 0.0, 1.0)
    return data, label

# Dataset API
dataset = tf.contrib.data.Dataset.from_tensor_slices((files))
dataset = dataset.map(parse1_func, num_threads=threads,
                      output_buffer_size=threads * 64)
dataset = dataset.map(lambda block: tuple(tf.py_func(parse2_pyfunc, [block], [tf.float32, tf.float32])),
                      num_threads=threads, output_buffer_size=threads * 64)
dataset = dataset.map(parse3_func, num_threads=threads,
                      output_buffer_size=threads * 64)
if shuffle: dataset = dataset.shuffle(buffer_size)
dataset = dataset.batch(batch_size)
dataset = dataset.repeat(num_epochs)
print(dataset.output_types, dataset.output_shapes)

iterator = dataset.make_one_shot_iterator()
next_data, next_label = iterator.get_next()

with tf.Session() as sess:
    #mkdir(save_path)
    while True:
        sess.run(next_data)
        gc.collect()
    #out_files = [os.path.join(save_path, '{:0>5}.png'.format(i)) for i in range(batch_size)]
    #ImageBatchWriter(sess, next_data, out_files)
