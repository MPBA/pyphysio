# coding=utf-8
from __future__ import division

import numpy as _np
from ..BaseIndicator import Indicator as _Indicator
from ..filters.Filters import Diff as _Diff
from ..Utility import PhUI as _PhUI
from ..Parameters import Parameter as _Par

__author__ = 'AleB'


class Histogram(_Indicator):
    @classmethod
    def algorithm(cls, signal, params):
        """
        Calculates the Histogram data to cache
        @return: (values, bins)
        @rtype: (array, array)
        """

        return _np.histogram(signal, params['histogram_bins'])

    _params_descriptors = {
        'histogram_bins': _Par(1, list,
                               'Number of bins (int) or bin edges, including the rightmost edge (list-like).', 100,
                               lambda x: type(x) is not int or x > 0)
    }


# """
# class HistogramMax(_Indicator):
# @classmethod
#     def algorithm(cls, signal, params):
#
# #        Calculates the Histogram's max value
# #        @return: (values, bins)
# #        @rtype: (array, array)
#
#         h, b = Histogram(params)(signal)
#         return _np.max(h)
#
#     @classmethod
#     def get_used_params(cls):
#         return Histogram.get_used_params()
# """


class Mean(_Indicator):
    """
    Compute the arithmetic mean along the specified axis, ignoring NaNs.

    Uses directly numpy.nanmean, but uses the PyPhysio cache.
    """

    @classmethod
    def algorithm(cls, data, params):
        return _np.nanmean(data)


class Min(_Indicator):
    """
    Return minimum of the data, ignoring any NaNs.

    Uses directly numpy.nanmin, but uses the PyPhysio cache.
    """

    @classmethod
    def algorithm(cls, data, params):
        return _np.nanmin(data)


class Max(_Indicator):
    """
    Return maximum of the data, ignoring any NaNs.

    Uses directly numpy.nanmax, but uses the PyPhysio cache.
    """

    @classmethod
    def algorithm(cls, data, params):
        return _np.nanmax(data)


class Range(_Indicator):
    """
    Computes the range value of the data, ignoring any NaNs.
    The range is the difference Max(d) - Min(d)
    """

    @classmethod
    def algorithm(cls, data, params):
        return Max()(data) - Min()(data)


class Median(_Indicator):
    """
    Computes the median of the data.

    Uses directly numpy.median but uses the PyPhysio cache.
    """

    @classmethod
    def algorithm(cls, data, params):
        return _np.median(data)


class StDev(_Indicator):
    """
    Computes the standard deviation of the data, ignoring any NaNs.

    Uses directly numpy.nanstd but uses the PyPhysio cache.
    """

    @classmethod
    def algorithm(cls, data, params):
        return _np.nanstd(data)


class Sum(_Indicator):
    """
    Computes the sum of the values in the data, treating Not a Numbers (NaNs) as zero.

    Uses directly numpy.nansum but uses the PyPhysio cache.
    """

    @classmethod
    def algorithm(cls, data, params):
        return _np.nansum(data)


class AUC(_Indicator):
    """
    Computes the Area Under the Curve of the data, treating Not a Numbers (NaNs) as zero.
    """

    @classmethod
    def algorithm(cls, data, params):
        fsamp = data.get_sampling_freq()
        return (1. / fsamp) * Sum()(data)


# HRV
class RMSSD(_Indicator):
    """
    Calculates the square root of the mean of the squared differences.
    """

    @classmethod
    def algorithm(cls, data, params):
        diff = _Diff()(data)
        return _np.sqrt(_np.sum(_np.power(diff, 2)) / len(diff))


class SDSD(_Indicator):
    """Calculates the standard deviation of the differences between each value and its next."""

    @classmethod
    def algorithm(cls, data, params):
        diff = _Diff()(data)
        return StDev()(diff)


class Triang(_Indicator):
    """Calculates the Triangular index."""

    @classmethod
    def algorithm(cls, data, params):
        min_ibi = _np.min(data)
        max_ibi = _np.max(data)
        bins = _np.arange(min_ibi, max_ibi, 1000. / 128)
        if len(bins) >= 10:
            h, b = Histogram(histogram_bins=bins)(data)
            return len(data) / _np.max(h)
        else:
            _PhUI.w("len(bins) < 10")
            return _np.nan


class TINN(_Indicator):
    """Calculates the difference between two histogram-related indicators."""

    @classmethod
    def algorithm(cls, data, params):
        min_ibi = _np.min(data)
        max_ibi = _np.max(data)
        bins = _np.arange(min_ibi, max_ibi, 1000. / 128)
        if len(bins) >= 10:
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
        else:
            _PhUI.w("len(bins) < 10")
            return _np.nan
