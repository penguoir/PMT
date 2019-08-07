from make_df0 import noise_filter
import numpy as np


def end_finder(signal, channel):
    points = signal[:, channel]
    noise_free = noise_filter(points)
    peaks = []
    max_length = 0
    end1, end2 = 0, 0
    length = 0
    for i in noise_free:
        peaks.append(points[i])
    for i in range(len(peaks)):
        if peaks[i+1] - peaks[i] == 1:
            length+=1
        else:
            if length > max_length:
                end2 = i
                end1 = i - length
                max_length = length
    median = np.median(points)
    while 0 <= end1 < len(points) - 1:
        if points[end1] > median:
            end1 += 1
        elif points[end1] < median:
            end1 -= 1
        else:
            break
    while end1 < end2 < len(points):
        if points[end2] < median:
            end2 += 1
        elif points[end2] > median:
            end2-=1
        else:
            break
    return end1, end2
