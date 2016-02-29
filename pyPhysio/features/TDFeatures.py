# coding=utf-8
from __future__ import division
from ..BaseFeature import Feature as _Feature
from ..features.SupportValues import SumSV as _SumSV, LengthSV as _LengthSV, DiffsSV as _DiffsSV, MedianSV as _MedianSV
from ..features.CacheOnlyFeatures import Histogram as _Histogram, HistogramMax as _HistogramMax
import numpy as _np

__author__ = 'AleB'


class TDFeature(_Feature):
    """
    This is the base class for the Time Domain Indexes.
    """

    def __init__(self, params=None, _kwargs=None):
        super(TDFeature, self).__init__(params, _kwargs)

    @classmethod
    def algorithm(cls, data, params):
        """
        Placeholder for the subclasses
        @raise NotImplementedError: Ever
        :param params:
        :param data:
        """
        raise NotImplementedError(cls.__name__ + " is a TDFeature but it is not implemented.")


class Mean(TDFeature):
    """
    Calculates the average value of the data.
    """

    def __init__(self, params=None, **kwargs):
        super(Mean, self).__init__(params, kwargs)

    @classmethod
    def algorithm(cls, data, params):
        return _np.mean(data)

    @classmethod
    def compute_on(cls, state):
        return state[_SumSV].value / float(state[_LengthSV].value)

    @classmethod
    def required_sv(cls):
        return [_SumSV, _LengthSV]


class Median(TDFeature):
    """
    Calculates the median of the data series.
    """

    def __init__(self, params=None, **kwargs):
        super(Median, self).__init__(params, kwargs)

    @classmethod
    def algorithm(cls, data, params):
        return _np.median(data)

    @classmethod
    def required_sv(cls):
        return [_MedianSV]

    @classmethod
    def compute_on(cls, state):
        return state[_MedianSV].value


class SD(TDFeature):
    """
    Calculates the standard deviation of the data series.
    """

    def __init__(self, params=None, **kwargs):
        super(SD, self).__init__(params, kwargs)

    @classmethod
    def algorithm(cls, data, params):
        return _np.std(data)


class PNNx(TDFeature):
    """
    Calculates the relative frequency (0.0-1.0) of pairs of consecutive IBIs in the data series
    where the difference between the two values is greater than the parameter (threshold).
    """

    def __init__(self, params=None, _kwargs=None, **kwargs):
        if type(_kwargs) is dict:
            kwargs.update(_kwargs)
        super(PNNx, self).__init__(params, kwargs)

    @classmethod
    def algorithm(cls, data, params):
        if cls == PNNx:
            assert 'threshold' in params, "Need the parameter 'threshold'."
            px = params
        else:
            px = params.copy()
            px.update({'threshold': cls.threshold()})
        return NNx.algorithm(data, px) / float(len(data))

    @staticmethod
    def threshold():
        raise NotImplementedError()

    @classmethod
    def required_sv(cls):
        return NNx.required_sv()

    @classmethod
    def compute_on(cls, state):
        return NNx.compute_on(state, cls.threshold()) / state[_LengthSV].value


from ..filters.Filters import Diff


class NNx(TDFeature):
    """
    Calculates number of pairs of consecutive values in the data where the difference between is greater than the given
    parameter (threshold).
    """

    def __init__(self, params=None, _kwargs=None, **kwargs):
        if type(_kwargs) is dict:
            kwargs.update(_kwargs)
        super(NNx, self).__init__(params, kwargs)

    @classmethod
    def algorithm(cls, data, params):
        if cls == NNx:
            assert 'threshold' in params, "Need the parameter 'threshold'."
            th = params['threshold']
        else:
            th = cls.threshold()
        diff = Diff.get(data)
        return sum(1.0 for x in diff if x > th)

    @staticmethod
    def get_used_params(**kwargs):
        return ['threshold']

    @staticmethod
    def threshold():
        raise NotImplementedError()

    @classmethod
    def required_sv(cls):
        return [_DiffsSV]

    @classmethod
    def compute_on(cls, state, threshold=None):
        if threshold is None:
            threshold = cls.threshold()
        return sum(1 for x in state[_DiffsSV].value if x > threshold)


class PNN10(PNNx):
    """
    Calculates the relative frequency (0.0-1.0) of the pairs of consecutive values in the data where the difference is
    greater than 10.
    """

    def __init__(self, params=None, **kwargs):
        super(PNN10, self).__init__(params, kwargs)

    @staticmethod
    def threshold():
        return 10


class PNN25(PNNx):
    """
    Calculates the relative frequency (0.0-1.0) of the pairs of consecutive values in the data where the difference is
    greater than 25.
    """

    def __init__(self, params=None, **kwargs):
        super(PNN25, self).__init__(params, kwargs)

    @staticmethod
    def threshold():
        return 25


class PNN50(PNNx):
    """
    Calculates the relative frequency (0.0-1.0) of the pairs of consecutive values in the data where the difference is
    greater than 50.
    """

    def __init__(self, params=None, **kwargs):
        super(PNN50, self).__init__(params, kwargs)

    @staticmethod
    def threshold():
        return 50


class NN10(NNx):
    """
    Calculates number of pairs of consecutive values in the data where the difference is greater than 10.
    """

    def __init__(self, params=None, **kwargs):
        super(NN10, self).__init__(params, kwargs)

    @staticmethod
    def threshold():
        return 10


class NN25(NNx):
    """
    Calculates number of pairs of consecutive values in the data where the difference is greater than 25.
    """

    def __init__(self, params=None, **kwargs):
        super(NN25, self).__init__(params, kwargs)

    @staticmethod
    def threshold():
        return 25


class NN50(NNx):
    """
    Calculates number of pairs of consecutive values in the data where the difference is greater than 50.
    """

    def __init__(self, params=None, **kwargs):
        super(NN50, self).__init__(params, kwargs)

    @staticmethod
    def threshold():
        return 50


class RMSSD(TDFeature):
    """
    Calculates the square root of the mean of the squared differences.
    """

    def __init__(self, params=None, **kwargs):
        super(RMSSD, self).__init__(params, kwargs)

    @classmethod
    def algorithm(cls, data, params):
        diff = Diff.get(data)
        return _np.sqrt(sum(diff ** 2) / len(diff))


class DiffSD(TDFeature):
    """Calculates the standard deviation of the differences between each value and its next."""

    def __init__(self, params=None, **kwargs):
        super(DiffSD, self).__init__(params, kwargs)

    @classmethod
    def algorithm(cls, data, params):
        diff = Diff.get(data)
        return _np.std(diff)


# TODO: fix documentation
class Triang(TDFeature):
    """Calculates the Triangular index that is the ratio between the number of samples and the number of samples in the
    highest bin of the data's 100 bin histogram."""

    def __init__(self, params=None, **kwargs):
        super(Triang, self).__init__(params, kwargs)

    @classmethod
    def algorithm(cls, data, params):
        h, b = _HistogramMax.get(data, histogram_bins=100)
        return len(data) / _np.max(h)  # TODO: check if the formula is the right one or use the HistogramMax


# TODO: fix documentation
class TINN(TDFeature):
    """Calculates the difference between two histogram-related features."""

    def __init__(self, params=None, **kwargs):
        super(TINN, self).__init__(params, kwargs)

    @classmethod
    def algorithm(cls, data, params):
        hist, bins = _Histogram.get(data, histogram_bins=100)
        max_x = _HistogramMax.get(data)
        hist_left = _np.array(hist[0:_np.argmax(hist)])
        ll = len(hist_left)
        hist_right = _np.array(hist[_np.argmax(hist):-1])
        rl = len(hist_right)

        y_left = _np.array(_np.linspace(0, max_x, ll))

        minx = _np.Inf
        pos = 0
        for i in range(len(hist_left) - 1):
            curr_min = _np.sum((hist_left - y_left) ** 2)
            if curr_min < minx:
                minx = curr_min
                pos = i
            y_left[i] = 0
            y_left[i + 1:] = _np.linspace(0, max_x, ll - i - 1)

        n = bins[pos - 1]

        y_right = _np.array(_np.linspace(max_x, 0, rl))
        minx = _np.Inf
        pos = 0
        for i in range(rl, 1, -1):
            curr_min = _np.sum((hist_right - y_right) ** 2)
            if curr_min < minx:
                minx = curr_min
                pos = i
            y_right[i - 1] = 0
            y_right[0:i - 2] = _np.linspace(max_x, 0, i - 2)

        m = bins[_np.argmax(hist) + pos + 1]

        return m - n
