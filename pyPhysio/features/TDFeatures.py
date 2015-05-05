from __future__ import division

__author__ = 'AleB'
__all__ = ['Mean', 'Median', 'SD', 'DiffSD', 'NNx', 'PNNx', 'NN10', 'NN25', 'NN50', 'PNN10', 'PNN25', 'PNN50', 'RMSSD',
           "Triang", "TINN"]

import numpy as np

from pyPhysio.features.BaseFeatures import TDFeature
from pyPhysio.features.SupportValues import SumSV, LengthSV, DiffsSV, MedianSV
from pyPhysio.features.CacheOnlyFeatures import Diff, Histogram, HistogramMax


class Mean(TDFeature):
    """
    Calculates the average value of the data.
    """

    def __init__(self, params=None, **kwargs):
        super(self.__class__, self).__init__(params, kwargs)

    @classmethod
    def raw_compute(cls, data, params):
        return np.mean(data)

    @classmethod
    def compute_on(cls, state):
        return state[SumSV].value / float(state[LengthSV].value)

    @classmethod
    def required_sv(cls):
        return [SumSV, LengthSV]


class Median(TDFeature):
    """
    Calculates the median of the data series.
    """

    def __init__(self, params=None, **kwargs):
        super(self.__class__, self).__init__(params, kwargs)

    @classmethod
    def raw_compute(cls, data, params):
        return np.median(data)

    @classmethod
    def required_sv(cls):
        return [MedianSV]

    @classmethod
    def compute_on(cls, state):
        return state[MedianSV].value


class SD(TDFeature):
    """
    Calculates the standard deviation of the data series.
    """

    def __init__(self, params=None, **kwargs):
        super(self.__class__, self).__init__(params, kwargs)

    @classmethod
    def raw_compute(cls, data, params):
        return np.std(data)


class PNNx(TDFeature):
    """
    Calculates the relative frequency (0.0-1.0) of pairs of consecutive IBIs in the data series
    where the difference between the two values is greater than the parameter (threshold).
    """

    def __init__(self, params=None, **kwargs):
        super(self.__class__, self).__init__(params, kwargs)

    @classmethod
    def raw_compute(cls, data, params):
        if cls == PNNx:
            assert 'threshold' in params, "Need the parameter 'threshold'."
            px = params
        else:
            px = params.copy().update({'threshold': cls.threshold()})
        return NNx.raw_compute(data, px) / float(len(data))

    @staticmethod
    def threshold():
        raise NotImplementedError()

    @classmethod
    def required_sv(cls):
        return NNx.required_sv()

    @classmethod
    def compute_on(cls, state):
        return NNx.compute_on(state, cls.threshold()) / state[LengthSV].value


class NNx(TDFeature):
    """
    Calculates number of pairs of consecutive values in the data where the difference between is greater than the given
    parameter (threshold).
    """

    def __init__(self, params=None, **kwargs):
        super(self.__class__, self).__init__(params, kwargs)

    @classmethod
    def raw_compute(cls, data, params):
        if cls == NNx:
            assert 'threshold' in params, "Need the parameter 'threshold'."
            th = params['threshold']
        else:
            th = cls.threshold()
        diff = Diff.get(data)
        return sum(1.0 for x in diff if x > th)

    @staticmethod
    def threshold():
        raise NotImplementedError()

    @classmethod
    def required_sv(cls):
        return [DiffsSV]

    @classmethod
    def compute_on(cls, state, threshold=None):
        if threshold is None:
            threshold = cls.threshold()
        return sum(1 for x in state[DiffsSV].value if x > threshold)


class PNN10(PNNx):
    """
    Calculates the relative frequency (0.0-1.0) of the pairs of consecutive values in the data where the difference is
    greater than 10.
    """

    def __init__(self, params=None, **kwargs):
        super(self.__class__, self).__init__(params, kwargs)

    @staticmethod
    def threshold():
        return 10


class PNN25(PNNx):
    """
    Calculates the relative frequency (0.0-1.0) of the pairs of consecutive values in the data where the difference is
    greater than 25.
    """

    def __init__(self, params=None, **kwargs):
        super(self.__class__, self).__init__(params, kwargs)

    @staticmethod
    def threshold():
        return 25


class PNN50(PNNx):
    """
    Calculates the relative frequency (0.0-1.0) of the pairs of consecutive values in the data where the difference is
    greater than 50.
    """

    def __init__(self, params=None, **kwargs):
        super(self.__class__, self).__init__(params, kwargs)

    @staticmethod
    def threshold():
        return 50


class NN10(NNx):
    """
    Calculates number of pairs of consecutive values in the data where the difference between is greater than 10.
    """

    def __init__(self, params=None, **kwargs):
        super(self.__class__, self).__init__(params, kwargs)

    @staticmethod
    def threshold():
        return 10


class NN25(NNx):
    """
    Calculates number of pairs of consecutive values in the data where the difference between is greater than 25.
    """

    def __init__(self, params=None, **kwargs):
        super(self.__class__, self).__init__(params, kwargs)

    @staticmethod
    def threshold():
        return 25


class NN50(NNx):
    """
    Calculates number of pairs of consecutive values in the data where the difference between is greater than 50.
    """

    def __init__(self, params=None, **kwargs):
        super(self.__class__, self).__init__(params, kwargs)

    @staticmethod
    def threshold():
        return 50


class RMSSD(TDFeature):
    """
    Calculates the square root of the mean of the squared differences.
    """

    def __init__(self, params=None, **kwargs):
        super(self.__class__, self).__init__(params, kwargs)

    @classmethod
    def raw_compute(cls, data, params):
        diff = Diff.get(data)
        return np.sqrt(sum(diff ** 2) / len(diff))


class DiffSD(TDFeature):
    """Calculates the standard deviation of the differences between each value and its next."""

    def __init__(self, params=None, **kwargs):
        super(self.__class__, self).__init__(params, kwargs)

    @classmethod
    def raw_compute(cls, data, params):
        diff = Diff.get(data)
        return np.std(diff)


# TODO: fix documentation
class Triang(TDFeature):
    """Calculates the Triangular index that is the ratio between the number of samples and the number of samples in the
    highest histogram bin of the data."""

    def __init__(self, params=None, **kwargs):
        super(self.__class__, self).__init__(params, kwargs)

    @classmethod
    def raw_compute(cls, data, params):
        h, b = Histogram.get(data)
        return len(data) / np.max(h)


# TODO: fix documentation
class TINN(TDFeature):
    """Calculates the difference between two histogram-related features."""

    def __init__(self, params=None, **kwargs):
        super(self.__class__, self).__init__(params, kwargs)

    @classmethod
    def raw_compute(cls, data, params):
        hist, bins = Histogram.get(data)
        max_x = HistogramMax.get(data)
        hist_left = np.array(hist[0:np.argmax(hist)])
        ll = len(hist_left)
        hist_right = np.array(hist[np.argmax(hist):-1])
        rl = len(hist_right)

        y_left = np.array(np.linspace(0, max_x, ll))

        minx = np.Inf
        pos = 0
        for i in range(len(hist_left) - 1):
            curr_min = np.sum((hist_left - y_left) ** 2)
            if curr_min < minx:
                minx = curr_min
                pos = i
            y_left[i] = 0
            y_left[i + 1:] = np.linspace(0, max_x, ll - i - 1)

        n = bins[pos - 1]

        y_right = np.array(np.linspace(max_x, 0, rl))
        minx = np.Inf
        pos = 0
        for i in range(rl, 1, -1):
            curr_min = np.sum((hist_right - y_right) ** 2)
            if curr_min < minx:
                minx = curr_min
                pos = i
            y_right[i - 1] = 0
            y_right[0:i - 2] = np.linspace(max_x, 0, i - 2)

        m = bins[np.argmax(hist) + pos + 1]

        return m - n
