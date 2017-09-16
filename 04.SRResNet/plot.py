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
            x, y = smooth(x, y)
            #ax.plot(x, y, label=labels[_])
            ax.plot(x, y, label=labels[_], marker=MARKERS[_], markersize=4)
        
        #ax.axis(ymin=0)
        ax.legend(loc=legend)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PATH, 'stats-{}.{}.png'.format(name if name else postfix, index)))
        plt.close() 
    
    _plot(2, 'MAD (RGB)', yscale='log')
    _plot(4, 'MS-SSIM (Y)', legend=2)

#plot1([120, 121, 122, 123], ['model=1 PReLU', 'model=2 PReLU', 'model=2 ReLU', 'model=3 ReLU'], '120')
#plot1([122, 127, 128, 129, 130], ['(model=2 ReLU)', 'batch_norm(0.9)', 'batch_norm(0.99)', 'batch_norm(0.9968)', 'batch_norm(0.999)'], '122.batch_norm')
plot1([122, 126, 132], ['(model=2 ReLU lr_min=4e-5)', 'init_activation=2.0', 'init_activation=2.0 lr_min=1e-6'], '122')
plot1([132, 134, 133], ['(model=2 ReLU init_activation=2.0 lr_min=1e-6)', 'batch_norm(0.968)', 'batch_norm(0.99)'], '132')
