import os, sys
import numpy as np
from scipy import signal
from itertools import product
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import time
import statistics as mean
import random
from fun import do_smd, do_dif, find_peaks, analize_peaks, fix_peaks
from itai_functions import end_finder


def noise_filter(events):
    mean = np.mean(events)
    max = np.amax(events)
    error = 2 * mean - max
    signals = list(range(len(events)))
    for i in range(len(signals)):
        if events[i] > error:
            signals.remove(i)
    m1 =[]
    for i in range(len(events)):
        m1.append(error)
    plt.plot(range(len(events)), m1, 'b-')
    plt.show()
    return signals


def gain(events, R, e, a):
    signals = noise_filter(events)
    area = 0
    for i in signals:
        area += events(i)
    gain = area / (e*a*R*f)
    return gain

def check_side(side_right, xm, y, ydata1, nfilter):
    side_width = 0
    if side_right:
        while True:
            if ydata1[xm] > nfilter:
                plt.plot(xm, ydata1[xm], 'b.')
                return side_width
            side_width = side_width+1
            xm = xm+1

    else:
        while True:
            if ydata1[xm] > ydata1[xm - 1]:
                plt.plot(xm, ydata1[xm], 'b.')
                return side_width
            side_width = side_width + 1
            xm = xm-1


def check_width(mx, my, ydata1):
    right_side = check_side(True, mx, my, ydata1)
    left_side = check_side(False, mx, my, ydata1)
    width = right_side+left_side
    return width


def filteron(n):
    n_filter= noise_filter(n)
    big_filter = n_filter
    for itera in range(len(n_filter)):
        init, fin = end_finder(n[n_filter[itera]])
        if (fin- init)<40:
            big_filter.remove()
    # check_width()
    for index in big_filter:
        use_x = [index]
        use_y = [n[index]]
        plt.plot(use_x,use_y, 'r+')
    plt.show()
    # plt.plot(x, y, '-')
    # plt.show()

#===========================================
#===========================================


v = 850
f = 5
wind = 15
PMT_num = 20
time_samples = 1024
start_time = time.time()
path = "C:\\Users\\Yanai's laptop\\Downloads\\"
file = open(path + 'out.DXD', 'rb')
event = 1
while event < 100000:
    if event % 10 == 0:
        print('Event number {} voltage: {}, frequency: {} wind {}, ({} files per sec).'.format(event, v, f, wind,
                                                                                               100 / (
                                                                                                           time.time() - start_time)))
        start_time = time.time()
    Data = np.fromfile(file, np.float32, (PMT_num + 4) * (time_samples + 2))
    if len(Data) < (PMT_num + 4) * (time_samples + 2):
        break
    Data = np.reshape(Data, (PMT_num + 4, time_samples + 2)).T
    data = Data[2:1002, :PMT_num]
    smd = do_smd(data, wind)
    print ('start\n\n\n\n')
    for i in range(20):
        plt.plot(range(len(smd[:, i])), smd[:, i], 'r-')
        x = [np.argmin(smd[:, i])]
        y = [smd[:, i][x]]
        plt.plot(x, y, '.')
        filteron(smd[:, i])
        #mn = [mean.mean(smd[:, i])]
        #for it in range(len(smd[:, i])-1):
        #    mn.append(mn[0])
        #plt.plot(range(len(smd[:, i])), mn )
        plt.show()

        print(smd[i])

    trig = np.argmax(np.roll(smd[:, 0], 1) - np.roll(smd[:, 0], -1))

#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------


