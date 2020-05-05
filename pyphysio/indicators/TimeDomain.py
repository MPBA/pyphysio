# coding=utf-8
from __future__ import division

import numpy as _np

from ..BaseIndicator import Indicator as _Indicator
from ..tools.Tools import Diff as _Diff
from ..Signal import EvenlySignal as _EvenlySignal, Signal as _Signal


__author__ = 'AleB'


class Mean(_Indicator):
    """
    Compute the arithmetic mean of the signal, ignoring any NaNs.
    """
    def __init__(self, **kwargs):
        _Indicator.__init__(self, **kwargs)

    @classmethod
    def algorithm(cls, data, params):
        return _np.nanmean(data.get_values())


class Min(_Indicator):
    """
    Return minimum of the signal, ignoring any NaNs.
    """
    def __init__(self, **kwargs):
        _Indicator.__init__(self, **kwargs)

    @classmethod
    def algorithm(cls, data, params):
        return _np.nanmin(data.get_values())


class Max(_Indicator):
    """
    Return maximum of the signal, ignoring any NaNs.
    """
    def __init__(self, **kwargs):
        _Indicator.__init__(self, **kwargs)

    @classmethod
    def algorithm(cls, data, params):
        return _np.nanmax(data.get_values())


class Range(_Indicator):
    """
    Compute the range of the signal, ignoring any NaNs.
    """
    def __init__(self, **kwargs):
        _Indicator.__init__(self, **kwargs)

    @classmethod
    def algorithm(cls, data, params):
        return Max()(data) - Min()(data)


class Median(_Indicator):
    """
    Compute the median of the signal, ignoring any NaNs.
    """
    def __init__(self, **kwargs):
        _Indicator.__init__(self, **kwargs)

    @classmethod
    def algorithm(cls, data, params):
        return _np.median(data.get_values())


class StDev(_Indicator):
    """
    Computes the standard deviation of the signal, ignoring any NaNs.
    """
    def __init__(self, **kwargs):
        _Indicator.__init__(self, **kwargs)

    @classmethod
    def algorithm(cls, data, params):
        return _np.nanstd(data.get_values())


class Sum(_Indicator):
    """
    Computes the sum of the values in the signal, treating Not a Numbers (NaNs) as zero.
    """
    def __init__(self, **kwargs):
        _Indicator.__init__(self, **kwargs)

    @classmethod
    def algorithm(cls, data, params):
        return _np.nansum(data.get_values())


class AUC(_Indicator):
    """
    Computes the Area Under the Curve of the signal, treating Not a Numbers (NaNs) as zero.
    """
    def __init__(self, **kwargs):
        _Indicator.__init__(self, **kwargs)

    @classmethod
    def algorithm(cls, signal, params):
        if isinstance(signal, _Signal) and not isinstance(signal, _EvenlySignal):
            cls.warn('Calculating Area Under the Curve of an Unevenly signal!')
        fsamp = signal.get_sampling_freq()
        return (1. / fsamp) * Sum()(signal)
    
class DetrendedAUC(_Indicator):
    """
    Computes the Area Under the Curve of the signal, treating Not a Numbers (NaNs) as zero.
    """
    def __init__(self, **kwargs):
        _Indicator.__init__(self, **kwargs)

    @classmethod
    def algorithm(cls, signal, params):
        if isinstance(signal, _Signal) and not isinstance(signal, _EvenlySignal):
            cls.warn('Calculating Area Under the Curve of an Unevenly signal!')
        fsamp = signal.get_sampling_freq()
        
        #detrend
        t_signal = signal.get_times()
        intercept = signal[0]
        coeff = (signal[-1] - signal[0]) / signal.get_duration()
        baseline = coeff*(t_signal - t_signal[0]) + intercept
        
        signal_ = signal - baseline
        return (1. / fsamp) * Sum()(signal_)


class RMSSD(_Indicator):
    """
    Compute the square root of the mean of the squared 1st order discrete differences.
    """
    def __init__(self, **kwargs):
        _Indicator.__init__(self, **kwargs)

    @classmethod
    def algorithm(cls, signal, params):
        diff = _Diff()(signal)
        return _np.sqrt(_np.mean(_np.power(diff.get_values(), 2)))


class SDSD(_Indicator):
    """
    Calculate the standard deviation of the 1st order discrete differences.
    """
    def __init__(self, **kwargs):
        _Indicator.__init__(self, **kwargs)

    @classmethod
    def algorithm(cls, signal, params):
        diff = _Diff()(signal)
        return StDev()(diff)

# TODO: FIX Histogram missing
class Triang(_Indicator):
    """
    Computes the HRV triangular index.
    """
    def __init__(self, **kwargs):
        _Indicator.__init__(self, **kwargs)

    @classmethod
    def algorithm(cls, data, params):
        step = 1000. / 128
        min_ibi = _np.min(data)
        max_ibi = _np.max(data)
        if (max_ibi - min_ibi) / step + 1 < 10:
            cls.warn("len(bins) < 10")
            return _np.nan
        else:
            bins = _np.arange(min_ibi, max_ibi, step)
            h, b = Histogram(histogram_bins=bins)(data)
            return len(data) / _np.max(h)


class TINN(_Indicator):
    """
    Computes the triangular interpolation of NN interval histogram.
    """
    def __init__(self, **kwargs):
        _Indicator.__init__(self, **kwargs)

    @classmethod
    def algorithm(cls, data, params):
        step = 1000. / 128
        min_ibi = _np.min(data)
        max_ibi = _np.max(data)
        if (max_ibi - min_ibi) / step + 1 < 10:
            cls.warn("len(bins) < 10")
            return _np.nan
        else:
            bins = _np.arange(min_ibi, max_ibi, step)
            h, b = Histogram(histogram_bins=bins)(data)
            max_h = _np.max(h)
            hist_left = _np.array(h[0:_np.argmax(h)])
            ll = len(hist_left)
            hist_right = _np.array(h[_np.argmax(h):])
            rl = len(hist_right)
            y_left = _np.array(_np.linspace(0, max_h, ll))

            minx = _np.Inf
            pos = 0
            for i in range(1, len(hist_left) - 1):
                curr_min = _np.sum((hist_left - y_left) ** 2)
                if curr_min < minx:
                    minx = curr_min
                    pos = i
                y_left[i] = 0
                y_left[i + 1:] = _np.linspace(0, max_h, ll - i - 1)

            n = b[pos - 1]

            y_right = _np.array(_np.linspace(max_h, 0, rl))
            minx = _np.Inf
            pos = 0
            for i in range(rl - 1, 1, -1):
                curr_min = _np.sum((hist_right - y_right) ** 2)
                if curr_min < minx:
                    minx = curr_min
                    pos = i
                y_right[i - 1] = 0
                y_right[0:i - 2] = _np.linspace(max_h, 0, i - 2)

            m = b[_np.argmax(h) + pos + 1]
            return m - n
