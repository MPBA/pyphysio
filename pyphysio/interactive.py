from __future__ import division

import matplotlib.pyplot as plt
import numpy as _np

from pyphysio import UnevenlySignal


def annotate_ecg(ecg, ibi):
    """
    Allow annotation of ECG files through minimal UI

    mouse click : update position
    w : search peak around position
    q : delete peak around position
    """

    ibi_times = ibi.get_times()
    curr_x = 0

    def onclick(event):
        global curr_x
        print('button = %d x = %f' % (event.button, event.xdata))
        curr_x = event.xdata
        return curr_x

    def onkey(event):
        global curr_x
        global ecg
        key = event.key
        if key == 'w':
            ecg_portion = ecg.segment_time(curr_x - 0.02, curr_x + 0.02)
            add_peak(ecg_portion)
        elif key == 'q':
            remove_peak(curr_x)

    def add_peak(ecg_portion):
        global ibi_times
        mx = _np.argmax(ecg_portion)
        t_cand = ecg_portion.get_times()[mx]
        print('will add: ' + str(t_cand))
        plt.vlines(t_cand, _np.min(ecg), _np.max(ecg), 'g')
        ibi_times = _np.append(ibi_times, t_cand)
        return ibi_times

    def remove_peak(instant):
        global ibi_times
        global ecg
        idx_nearest = _np.argmin(abs(ibi_times - instant))
        t_nearest = ibi_times[idx_nearest]
        print('will remove: ' + str(t_nearest))
        plt.vlines(t_nearest, _np.min(ecg), _np.max(ecg), 'r')
        ibi_times = _np.delete(ibi_times, idx_nearest)
        return ibi_times

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ecg.get_times(), ecg.get_values())
    ax.vlines(ibi.get_times(), _np.min(ecg), _np.max(ecg))

    fig2 = plt.figure()
    plt.subplot(111, sharex=ax)
    ibi.plot()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    cid2 = fig.canvas.mpl_connect('key_press_event', onkey)

    ##########
    ##########

    ibi_times_sorted = _np.sort(ibi_times)

    ibi_values = _np.diff(ibi_times_sorted)
    ibi_values = _np.r_[ibi_values[0], ibi_values]
    ibi_ok = UnevenlySignal(ibi_values, ibi.get_sampling_freq(), 'IBI', x_values=ibi_times_sorted, x_type='instants')

    return ibi_ok
