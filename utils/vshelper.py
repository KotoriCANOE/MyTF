import numpy as np
import vapoursynth as vs

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
