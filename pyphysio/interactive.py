# coding=utf-8
from __future__ import print_function
from __future__ import division

import matplotlib.pyplot as plt
import pyphysio as ph
import numpy as np


class _MouseSelectionFilter(object):
    def __init__(self, onselect):
        self._select = onselect
        self._last_press = None

    def on_move(self, event):
        self._last_press = None

    def on_press(self, event):
        x, y = event.xdata, event.ydata
        self._last_press = x, y, event.button

    def on_release(self, event):
        x, y = event.xdata, event.ydata
        if self._last_press is not None:
            xx, yy, b = self._last_press
            if x == xx and y == yy and event.button == b:
                self._select(event)


class _ItemManager(object):
    def __init__(self, snap_func, select, unselect, add, delete):
        self._snap_func = snap_func
        self._select = select
        self._unselect = unselect
        self._delete = delete
        self._add = add
        self.selection = -1

    def unselect(self):
        self._unselect(self.selection)
        self.selection = None

    def on_select(self, ev):
        if ev.xdata is not None and ev.ydata is not None:
            x, y, item, new = self._snap_func(ev.xdata, ev.ydata)
#            print("on_select: %d, %d: %d" % (x, y, item))
            if self.selection is not None:
                self.unselect()
            if ev.button == 1:
                if new:
                    self._add(x, y, item)
                else:
                    self.selection = item
                    self._select(item)


class Annotate(object):
    def __init__(self, ecg, ibi):
        self.plots = None
        self.peaks_t = None
        self.done = False
        self.ecg = ecg
        self.ibi = ibi
        self.fig = plt.figure()
        self.p_sig = self.fig.add_subplot(2, 1, 1)
        self.p_res = self.fig.add_subplot(2, 1, 2, sharex=self.p_sig)

        self.max = ph.Max()(self.ecg)
        self.min = ph.Min()(self.ecg)

        self.margin = ph.Range()(self.ecg) * .1
        self.max += self.margin
        self.min -= self.margin

        if isinstance(ibi, ph.UnevenlySignal):
            self.peaks_t = self.ibi.get_times()
            self.peaks_v = self.ibi.get_values()
        else:
            self.peaks_t = np.empty(0)
            self.peaks_v = np.empty(0)

        self.p_sig.plot(self.ecg.get_times(), self.ecg.get_values(), 'b')

        self.p_res.plot(self.peaks_t, self.peaks_v, 'b'),
        self.p_res.plot(self.peaks_t, self.peaks_v, 'go')

        self.replot()

        class Cursor(object):
            left = None
            right = None
            radius = .3
            radiusi = int(radius * self.ecg.get_sampling_freq())

            @staticmethod
            def on_move(event):
                Cursor.draw(event)

            @staticmethod
            def on_scroll(event):
                if event.button == "up":
                    Cursor.radiusi += 3
                elif event.button == "down":
                    Cursor.radiusi -= 7
                Cursor.radius = Cursor.radiusi / self.ecg.get_sampling_freq()
                Cursor.draw(event)

            @staticmethod
            def draw(event):
                if Cursor.left is not None:
                    Cursor.left.remove()
                    Cursor.right.remove()
                    Cursor.left = None
                    Cursor.right = None
                if event.xdata is not None:  # TODO (Andrea): not do this if speed (dxdata/dt) is high
                    Cursor.left = self.p_sig.vlines(event.xdata - Cursor.radius, self.min - self.margin * 2,
                                                    self.max + self.margin * 2, 'k')
                    Cursor.right = self.p_sig.vlines(event.xdata + Cursor.radius, self.min - self.margin * 2,
                                                     self.max + self.margin * 2, 'k')
                self.fig.canvas.draw()

        def find_peak(s):
            return np.argmax(s)

        def snap(xdata, ydata):
            nearest_after = self.peaks_t.searchsorted(xdata)
            nearest_prev = nearest_after - 1

            dist_after = self.peaks_t[nearest_after] - xdata if 0 <= nearest_after < len(self.peaks_t) else None
            dist_prev = xdata - self.peaks_t[nearest_prev] if 0 <= nearest_prev < len(self.peaks_t) else None

            if dist_after is None or dist_prev < dist_after:
                if dist_prev is not None and dist_prev < Cursor.radius:
                    return self.peaks_t[nearest_prev], ydata, nearest_prev, False
            elif dist_prev is None or dist_after < dist_prev:
                if dist_after is not None and dist_after < Cursor.radius:
                    return self.peaks_t[nearest_after], ydata, nearest_after, False

            s = self.ecg.segment_time(xdata - Cursor.radius, xdata + Cursor.radius)
            s = np.array(s)
            m = find_peak(s)
            return xdata - Cursor.radius + m / self.ecg.get_sampling_freq(), ydata, nearest_after, True

        class Selector(object):
            selector = None

            @staticmethod
            def select(item):
#                print("select: %d" % item)
                Selector.selector = self.p_sig.vlines(self.peaks_t[item], self.min - self.margin, self.max + self.margin, 'g')

            @staticmethod
            def unselect(item):
                if Selector.selector is not None:
#                    print("unselect: %d" % item)
                    Selector.selector.remove()

        # it is correct that the computation of the values is done at the end (line 186)
        def add(time, y, pos):
            self.peaks_t = np.insert(self.peaks_t, pos, time)
            self.replot()

        def delete(item):
            self.peaks_t = np.delete(self.peaks_t, item)
            self.replot()

        im = _ItemManager(snap, Selector.select, Selector.unselect, add, delete)
        mf = _MouseSelectionFilter(im.on_select)

        def press(ev):
#            print(ev.key)
            if ev.key == "d" and im.selection is not None:
                delete(im.selection)
                im.unselect()
                
        def handle_close(ev):
            self.done = True
            return
                

            
        clim = self.fig.canvas.mpl_connect('motion_notify_event', lambda e: (mf.on_move(e), Cursor.on_move(e)))
        clip = self.fig.canvas.mpl_connect('button_press_event', mf.on_press)
        clir = self.fig.canvas.mpl_connect('button_release_event', mf.on_release)
        clis = self.fig.canvas.mpl_connect('scroll_event', Cursor.on_scroll)
        clik = self.fig.canvas.mpl_connect('key_press_event', press)
        ccls = self.fig.canvas.mpl_connect('close_event', handle_close)
        
        while not self.done :
#            print('waiting')
            plt.pause(1)
        
        plt.close(self.fig)
        # it is correct that the computation of the values is done at the end!
        # do not change!
        self.peaks_v = np.diff(self.peaks_t)
        self.peaks_v = np.r_[self.peaks_v[0], self.peaks_v]
                    
        if isinstance(ibi, ph.UnevenlySignal):
            self.ibi_ok =  ph.UnevenlySignal(values=self.peaks_v,
                                     sampling_freq=self.ibi.get_sampling_freq(),
                                     signal_type=self.ibi.get_signal_type(),
                                     start_time=self.ibi.get_start_time(),
                                     x_values=self.peaks_t,
                                     x_type='instants',
                                     duration=self.ibi.get_duration())
        else:
            self.ibi_ok = ph.UnevenlySignal(values=self.peaks_v,
                                     sampling_freq=self.ecg.get_sampling_freq(),
                                     signal_type=self.ecg.get_signal_type(),
                                     start_time=self.ecg.get_start_time(),
                                     x_values=self.peaks_t,
                                     x_type='instants',
                                     duration=self.ecg.get_duration())
        
    def __call__(self):
        return self.ibi_ok
    
    def replot(self):
        if self.plots is not None:
            self.plots.remove()
        if self.peaks_t is not None:
            self.plots = self.p_sig.vlines(self.peaks_t, self.min, self.max, 'y')
            self.fig.canvas.draw()
