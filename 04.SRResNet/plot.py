import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.interpolate import spline

PATH = 'plot'
if not os.path.exists(PATH): os.makedirs(PATH)

def smooth(x, y, sigma=1.0):
    y_new = ndimage.gaussian_filter1d(y, sigma, mode='reflect')
    return x, y_new

def plot1(postfix, labels=None, name=None):
    MARKERS = 'x+^vDsoph*'

    if labels is None:
        labels = [str(p) for p in postfix]
    else:
        for _ in range(len(postfix)):
            if _ < len(labels):
                if labels[_] is None:
                    labels[_] = str(postfix[_])
            else:
                labels.append(str(postfix[_]))
    
    def _plot(index, loss, legend=1, xscale='linear', yscale='linear', xfunc=None, yfunc=None):
        fig, ax = plt.subplots()
        
        xlabel = 'training steps'
        if xscale != 'linear': xlabel += ' ({} scale)'.format(xscale)
        ylabel = '{}'.format(loss)
        if yscale != 'linear': ylabel += ' ({} scale)'.format(yscale)
        
        ax.set_title('Test Error with Training Progress')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        
        for _ in range(len(postfix)):
            stats = np.load('test{}.tmp/stats.npy'.format(postfix[_]))
            if stats.shape[1] <= index:
                print('test{} doesn\'t have index={}'.format(index))
                continue
            stats = stats[1:]
            x = stats[:, 0]
            y = stats[:, index]
            if xfunc: x = xfunc(x)
            if yfunc: y = yfunc(y)
            #x, y = smooth(x, y)
            ax.plot(x, y, label=labels[_])
            #ax.plot(x, y, label=labels[_], marker=MARKERS[_], markersize=4)
        
        #ax.axis(ymin=0)
        ax.legend(loc=legend)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PATH, 'stats-{}.{}.png'.format(name if name else postfix, index)))
        plt.close()
    
    _plot(2, 'MAD (RGB)', yscale='log')
    _plot(4, 'MS-SSIM (Y)', legend=4)
    _plot(5, 'weighted loss', yscale='log')

#plot1([120, 121, 122, 123], ['model=1 PReLU', 'model=2 PReLU', 'model=2 ReLU', 'model=3 ReLU'], '120')
#plot1([122, 126, 132], ['(model=2 ReLU lr_min=4e-5)', 'init_activation=2.0', 'init_activation=2.0 lr_min=1e-6'], '122')
#plot1([132, 134, 133, 135], ['batch_norm(0)', 'batch_norm(0.968)', 'batch_norm(0.99)', 'batch_norm(0.9968)'], '132')
#plot1([136, 138, 133, 139, 137], ['weight_decay=0', 'weight_decay=1e-6', 'weight_decay=2e-6', 'weight_decay=5e-6', 'weight_decay=1e-5'], '136')
#plot1([140, 141, 133], ['initializer=3', 'initializer=4', 'initializer=5'], '140')
#plot1([141, 149, 142, 143, 144, 150, 151], ['ReLU', 'LReLU(0.05)', 'LReLU(0.3)', 'ELU', 'PReLU', 'BN+SU', 'SU'], '141')
#plot1([146, 145, 141, 147, 148], ['lr=5e-3', 'lr=2e-3', 'lr=1e-3', 'lr=5e-4', 'lr=2e-4'], '146')
#plot1([150, 153, 154, 155, 156], ['act+3conv+act+3conv', '3conv+act+3conv', 'act+1conv+act+3conv', 'channels2=64', 'final residual'], '153')
#plot1([156, 157, 159], ['final residual', 'resize conv as last', 'g_depth=16'], '157')
#plot1([156, 158, 160, 161], ['channels=64', 'channels=80', 'channels=96', 'channels=128'], '160')
#plot1(['158.1', '158.2'], ['lr_min=0', 'lr_min=5e-5'], '158')
#plot1(['158.1', '162.1'], ['Adam', 'Nadam'], '162')
#plot1([164, '162.1', 163, 165, 168], ['decay_step=250', 'decay_step=500', 'decay_step=1000', 'polynomial8', 'custom1'], '164')
#plot1([165, 166, 167], ['epsilon=1e-8', 'epsilon=1e-3', 'epsilon=1e-1'], '165')
#plot1(['162.1', 169, 170], ['random_resizer+noise+JPEG', 'random_resizer', 'spline16'], '169')
#plot1([170, 171, 172], ['spline16', 'DIV2K spline16', 'DIV2K Catmull-Rom'], '171')
#plot1(['162.1', 173, 174, 175], ['old dataset (shuffle=65536)', 'new dataset (shuffle=256)', 'new dataset (shuffle=262144)', 'new dataset (no split, shuffle=65536)'], '173')
#plot1(['162.1', 176, 177, 178], ['old train_ema=0', 'train_ema=0.9999', 'train_ema=0.999', 'train_ema=0'], '177')
#plot1(['162.1', 176, 179], ['old Nadam train_ema=0', 'Nadam train_ema=0.9999', 'Adam train_ema=0.9999'], '179')
#plot1(['162.1', 176, 180, 181], ['train_ema=0', 'train_ema=0.9999', 'adaptive decay 1, 0.50', 'adaptive decay 2, 0.50'], '180')
#plot1(['162.1', 176, 181, 182, 183], ['train_ema=0', 'train_ema=0.9999', 'adaptive decay 2, 0.50', 'adaptive decay 2, 0.29', 'adaptive decay 2, 0.16'], '181')
#plot1(['162.1', 176, 182, 184], ['Nadam train_ema=0', 'Nadam train_ema=0.9999', 'Nadam, adaptive decay 2, 0.29', 'Momentum, adaptive decay 2, 0.29'], '184')
#plot1(['162.1', 176, 182, 185], ['train_ema=0', 'train_ema=0.9999', 'adaptive decay 2, 0.29', 'adaptive decay 2, 0.29'], '185')
#plot1(['162.1', 182, 186], ['train_ema=0', 'adaptive decay 2, 0.29', 'SENet'], '186')
#plot1([186, 187], ['SENet', 'adaptive decay 3'], '187')
#plot1([186, 188, 189, 190], ['SENet', 'SU: double conv', 'SE: add resize conv', 'SENet v2'], '188')
#plot1([186, 190, 191, 192], ['SENet', 'SENet v2', 'sub-pixel conv', 'sub-pixel conv in skip0'], '190')
#plot1([192, 193, 194, 195], ['sub-pixel conv in skip0', 'no SE in skip1', 'SE: channel_r=1', 'SE: conv2d ver.'], '193')
#plot1([186, 194, 196, 197, 198], ['SENet', 'SE: channel_r=1', 'adaptive lr v3 10 6*0.8', 'adaptive lr v3 10 5*0.9', 'adaptive lr v3 20 6*0.9'], '196')
#plot1([194, 197, 198, 199, 200], ['SE: channel_r=1', 'adaptive lr v3 10 5*0.9', 'adaptive lr v3 20 6*0.9', 'adaptive lr v3 10 6*0.95', 'adaptive lr v4 10 5*0.9=>0.95'], '199')
#plot1([199, 201, 202], ['Pre: Medium', 'Pre: Heavy (from Medium)', 'Pre: Heavy (TF 1.4)'], '202')
#plot1([202, 203, 204, 205, 206, 207], ['subpixel conv: 3+5', 'subpixel conv: 4+4', 'deconv: 4+4', 'resize deconv: 4+4', 'resize conv: 4+4', 'resize conv: 3+5'], '203')
#plot1([208, 209, 207, 210, 211], ['resize conv: 3+3', 'resize conv: 3+4', 'resize conv: 3+5', 'resize conv: 3+7', 'resize conv: 3+5'], '207')
#plot1([207, 220, 221], ['TF 1.4, ms=1, c80 b8', 'TF 1.5, ms=2, c64 b8', 'TF 1.5, ms=2, c32 b16'], '220')
#plot1([207, 220, 222], ['TF 1.4, ms=1, c80 b8', 'TF 1.5, ms=2, c64 b8', 'TF 1.5, ms=1, c64 b8'], '222')
#plot1([207, 222, 223, 224, 225], ['TF 1.4, ms=1, c80 b8', 'TF 1.5, ms=1, c64 b8', 'TF 1.7, ms=1, c64 b8', 'TF 1.7, ms=1, c64 b16', 'TF 1.7, ms=0, c64 b8'], '223')



#plot1([1000, 1001], ['patch=96 batch=16', 'patch=64 batch=16'], '1000')
#plot1([1001, 1002, 1003], ['original resize, multi=3', 'with artifacts resize, multi=2', 'reduce artifacts, use_se=0, predown=True'], '1002')
#plot1([1003, 1004, 1005, 1006, 1007], ['use_se=0, predown=True', 'use_se=1, predown=False', 'use_se=2, predown=False', 'use_se=0, predown=False', 'use_se=1, predown=False, channels=64'], '1003')
#plot1([1001, 1007, 1008, 1009], ['original resize, multi=3', 'channels=64, mod resize 3, multi=2', 'channels=64, mod resize 4, multi=2', 'channels=64, mod resize 5, multi=2'], '1007')
#plot1([1009, 1010, 1011, 1012], [], '1009')
#plot1([1012, 1013, 1014, 1015, 1016], [], '1012')
#plot1([1014, 1016, 1020, 1021], [], '1020')
#plot1([1014, 1020, 1022, 1023, 1024], [], '1022')
#plot1([1023, 1025], [], '1023')
#plot1([1025, 1026, 1027, 1028], [], '1025')
#plot1([1029, 1030], [], '1029')
#plot1([1031, 1032, 1033, 1034, 1035], [], '1031')
#plot1([1025, 1032, 1034], [], '1034')
#plot1([1032, 1036, 1037], [], '1036')
#plot1([1036, 1038, 1039], [], '1038')
#plot1([1036, 1040, 1041, 1042], [], '1040')
#plot1([1036, 1043, 1044], [], '1043')
#plot1([1036, 1045, 1046], [], '1045')
#plot1([1047, 1048, 1049, 1050], [], '1047')
#plot1([1051, 1052, 1053], [], '1051')
#plot1([1014, 1020, 1045, 1054], [], '1054')
#plot1([1016, 1021, 1055], [], '1055')
#plot1([1049, 1060, 1061, 1062], [], '1060')
#plot1([1061, 1063, 1064, 1065, 1066], [], '1063')
#plot1([1061, 1064, 1065, 1067, 1068, 1069], [], '1065')
#plot1([1065, 1067, 1070, 1071], [], '1067')
#plot1([1054, 1055, 1072], [], '1072')
#plot1([1049, 1061, 1063, 1065, 1067], [], '1067.2')
#plot1([1049, 1065, 1067, 1073], [], '1073')
#plot1([1073, 1074, 1075, 1076, 1077, 1078], [], '1074')
#plot1([1049, 1065, 1073, 1079, 1080], [], '1079')
#plot1([1049, 1065, 1080, 1081], [], '1081')
#plot1([1080, 1081, 1082], [], '1082')
#plot1([1081, 1083, 1084, 1085], [], '1083')
#plot1([1086, 1087, 1088], [], '1086')
#plot1([1020, 1054, 1072, 1089], [], '1089')
#plot1([1090, 1091, 1092, 1093], [], '1090')
#plot1([1091, 1094, 1095, 1096, 1097], [], '1094')
#plot1([1091, 1097, 1098, 1099, 1100], [], '1097')
plot1([1097, 1101, 1102, 1103], [], '1101')


