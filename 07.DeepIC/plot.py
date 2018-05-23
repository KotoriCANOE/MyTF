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

#plot1([3, 4, 5, 6], ['ReLU', 'SU', 'Swish', 'Swish+SE'], '4')
#plot1([6, 7, 8], ['channels=64', 'channels=32', 'channels=48'], '7')
#plot1([8, 9, 10, 11], ['channels=48', 'batch_norm=0', 'k_first=7, k_last=7', 'd_depth=8'], '9')
#plot1([8, 12, 13, 14], ['activation=None', 'tanh', 'clip(-1, 1)', 'clip_swish(-1, 1)'], '12')
#plot1([12, 15], ['tanh', 'quantization'], '15')
#plot1([15, 16, 17], ['downscale=4', 'downscale=2', 'downscale=1'], '16')
#plot1([17, 18, 19], ['downscale=1', 'entropy loss 1', 'entropy loss 2'], '18')
#plot1([17, 20, 21, 22], ['downscale=1', 'PNG discriminator loss', 'PNG discriminator loss', 'PNG discriminator loss - binarized'], '20')
#plot1([17, 23, 24, 25], ['downscale=1', 'binarized enc as generator output', 'weight3=0.1', 'add ceiling before debinarization'], '23')
#plot1([17, 25, 26, 27], ['downscale=1', 'add ceiling before debinarization', 'L2 norm loss', 'average L2 norm loss'], '26')
#plot1([27, 28, 29, 30, 31], ['average L2 norm loss', 'MSE loss, binary=1.0, comp=0.05', 'MSE loss, binary=0.5, comp=0.1', 'fixed wrong skip in decoder', 'L1 norm loss'], '28')
plot1([30, 32], ['fixed wrong skip in decoder', 'correct clip'], '32')

