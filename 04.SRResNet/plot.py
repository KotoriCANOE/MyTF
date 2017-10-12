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

    if labels is None: labels = postfix
    
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
plot1([186, 187], ['SENet', 'adaptive decay 3'], '187')
