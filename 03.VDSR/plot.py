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
    _plot(4, 'MS-SSIM (Y)', legend=2)

#plot1([1, 2], ['batch_norm(0)', 'batch_norm(0.99)'], '2')
#plot1([3, 1, 4], ['relu', 'lrelu0.05', 'prelu'], '3')
#plot1([6, 3, 5, 7], ['lr=1e-2', 'lr=1e-1', 'lr=1e-1 decay=200', 'lr=1'], '6')
#plot1([7, 8], ['relu', 'su'], '7')
#plot1([8, 9, 10, 11, 12, 13], ['decay=100', 'decay=100', 'decay=100', 'decay=200', 'decay=200', 'decay=500'], '9')
#plot1([11, 14, 15], ['gclip=2e-3', 'gclip=1e-3', 'gclip=5e-4'], '11')
plot1([14], ['14'], '14')
