import matplotlib.pyplot as plt
import numpy as np
import re

def plot_log(filename):
    """Plot logs saved by train_ssd.py"""
    train_ce = []
    train_smoothl1 = []
    mAP = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            if 'Training cost' in line:
                res = re.findall("\d+\.\d+", line)
                assert len(res) == 4
                train_ce.append(float(res[1]))
                train_smoothl1.append(float(res[2]))
            elif 'mAP' in line:
                res = re.findall("\d+\.\d+", line)
                mAP.append(float(res[0]))
    xticks = np.arange(len(train_ce))
    fig, ax = plt.subplots(2, 1, sharex='col')
    ax[0].plot(xticks, train_ce, label='CrossEntropy')
    ax[0].plot(xticks, train_smoothl1, label='SmoothL1')
    ax[0].grid()
    ax[0].legend()
    ax[1].plot(xticks, mAP, label='validation mAP')
    plt.grid()
    plt.legend()
    plt.xlabel('Epoch')
    plt.show()
