import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.interpolate import spline

NUM_LABELS = 12
PATH = 'plot'
if not os.path.exists(PATH): os.makedirs(PATH)

def smooth(x, y, sigma=3.0):
    y_new = ndimage.gaussian_filter1d(y, sigma, mode='reflect')
    return x, y_new

def plot1(postfix, labels=None, name=None):
    MARKERS = 'x+^vDsoph*'

    if labels is None: labels = postfix
    fig, ax = plt.subplots()
    
    ax.set_title('Test Error with Training Progress')
    ax.set_xlabel('training iterations')
    ax.set_ylabel('mean absolute difference (log scale)')
    ax.set_xscale('linear')
    ax.set_yscale('log')

    for _ in range(len(postfix)):
        stats = np.load('test{}_{}.tmp/stats.npy'.format(NUM_LABELS, postfix[_]))
        stats = stats[1:]
        x, y = smooth(stats[:, 0], stats[:, 2])
        #ax.plot(x, y, label=labels[_])
        ax.plot(x, y, label=labels[_], marker=MARKERS[_], markersize=4)

    #ax.axis(ymin=0)
    ax.legend(loc=1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PATH, 'stats-{}.png'.format(name if name else postfix)))
    plt.close() 

plot1([102, 101, 100], ['weight decay: 0', 'weight decay: 1e-6', 'weight decay: 1e-5'], 'weight_decay')
plot1([104, 103, 100], ['first conv: 1x3', 'first conv: 1x5', 'first conv: 1x7'], 'k_first')
plot1([105, 106, 100, 107, 108], ['initializer: 1', 'initializer: 2', 'initializer: 3', 'initializer: 4', 'initializer: 5'], 'initializer')
plot1([110, 100, 109], ['channels: 32', 'channels: 48', 'channels: 64'], 'channels')
plot1([100, 113], ['0.9,0.999', '0.5,0.9'], 'learning_beta')
plot1([100, 112], ['learning rate: 1e-4', 'learning rate: 1e-3'], 'learning_rate')
plot1([116, 114, 115, 112], ['smooth: no, noise: no', 'smooth: yes, noise: no', 'smooth: no, noise: yes', 'smooth: yes, noise: yes'], 'data_pp')
plot1([119, 118, 117, 112], ['trainset size: 1000', 'trainset size: 10000', 'trainset size: 100000', 'trainset size: 1000000'], 'epoch_size')
plot1([121, 112, 120], ['trainset 1', 'trainset 2', 'trainset 1+2'], 'Test2')
'''
plot1([121, 112, 120], ['trainset 1', 'trainset 2', 'trainset 1+2'], 'Test1')
'''
