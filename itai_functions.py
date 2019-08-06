from make_df0 import noise_filter
import numpy as np


def end_finder(signal, channel):
    points = signal[:, channel]
    noise_free = noise_filter(points)
    peaks = []
    for i in noise_free:
        peaks.append(points[i])
    end1, end2 = np.argmin(peaks)
    median = np.median(points);
    while end1 > median:
        end1 -= 1
    while end2 < median:
        end2 += 1
    return end1, end2
