__author__ = 'AleB'
__all__ = ['Mean', 'Median', 'SD', 'SDSD', 'NNx', 'PNNx', 'NN10', 'NN25', 'NN50', 'PNN10', 'PNN25', 'PNN50', 'RMSSD',
           "Triang", "TINN"]

import numpy as np

from pyHRV.indexes.CacheOnlyFeatures import Diff, Histogram, HistogramMax
from pyHRV.indexes.BaseFeatures import TDFeature
from pyHRV.PyHRVSettings import MainSettings as Sett
from pyHRV.indexes.SupportValues import SumSV, LengthSV, DiffsSV, MedianSV


class Mean(TDFeature):
    """
    Calculates the average value of the data.
    """
    # TODO: WA a weighted average using the time spans?

    def __init__(self, data=None):
        super(Mean, self).__init__(data)
        self._value = Mean.get(self._data)

    @classmethod
    def _compute(cls, data, params):
        return np.mean(data)

    @classmethod
    def required_sv(cls):
        return [SumSV, LengthSV]

    @classmethod
    def calculate_on(cls, state):
        return state[SumSV].value / float(state[LengthSV].value)


class Median(TDFeature):
    """
    Calculates the median of the data series.
    """

    def __init__(self, data=None):
        super(Median, self).__init__(data)
        self._value = Median.get(self._data)

    @classmethod
    def _compute(cls, data, params):
        return np.median(data)

    @classmethod
    def required_sv(cls):
        return [MedianSV]

    @classmethod
    def calculate_on(cls, state):
        return state[MedianSV].value


class SD(TDFeature):
    """
    Calculates the standard deviation of the data series.
    """

    def __init__(self, data=None):
        super(SD, self).__init__(data)
        self._value = SD.get(self._data)

    @classmethod
    def _compute(cls, data, params):
        return np.std(data)


class PNNx(TDFeature):
    """
    Calculates the presence proportion (0.0-1.0) of pairs of consecutive IBIs in the data series
    where the difference between the two values is greater than the parameter (threshold).
    """

    def __init__(self, data=None, params=None):
        super(PNNx, self).__init__(data, params)
        self._xth = self._params['threshold'] \
            if self.__class__ != PNNx and self._params is not None and 'threshold' in self._params else self.threshold()
        self._value = NNx(data, self._xth).value / float(len(data))

    @staticmethod
    def threshold():
        return Sett.nnx_default_threshold

    @classmethod
    def required_sv(cls):
        return NNx.required_sv()

    @classmethod
    def calculate_on(cls, state):
        return NNx.calculate_on(state, cls.threshold()) / state[LengthSV].value


class NNx(TDFeature):
    """
    Calculates number of pairs of consecutive values in the data where the difference between is greater than the given
    parameter (threshold).
    """

    def __init__(self, data=None, params=None):
        super(NNx, self).__init__(data, params)
        self._xth = self._params['threshold'] \
            if self.__class__ != PNNx and self._params is not None and 'threshold' in self._params else self.threshold()
        diff = Diff.get(self._data)
        self._value = sum(1.0 for x in diff if x > self._xth)

    @staticmethod
    def threshold():
        return Sett.nnx_default_threshold

    @classmethod
    def required_sv(cls):
        return [DiffsSV]

    @classmethod
    def calculate_on(cls, state, threshold=None):
        if threshold is None:
            threshold = cls.threshold()

        return sum(1 for x in state[DiffsSV].value if x > threshold)


class PNN10(PNNx):
    """
    Calculates the relative frequency (0.0-1.0) of the pairs of consecutive values in the data where the difference is
    greater than 10.
    """

    @staticmethod
    def threshold():
        return 10


class PNN25(PNNx):
    """
    Calculates the relative frequency (0.0-1.0) of the pairs of consecutive values in the data where the difference is
    greater than 25.
    """

    @staticmethod
    def threshold():
        return 25


class PNN50(PNNx):
    """
    Calculates the relative frequency (0.0-1.0) of the pairs of consecutive values in the data where the difference is
    greater than 50.
    """

    @staticmethod
    def threshold():
        return 50


class NN10(NNx):
    """
    Calculates number of pairs of consecutive values in the data where the difference between is greater than 10.
    """

    @staticmethod
    def threshold():
        return 10


class NN25(NNx):
    """
    Calculates number of pairs of consecutive values in the data where the difference between is greater than 25.
    """

    @staticmethod
    def threshold():
        return 25


class NN50(NNx):
    """
    Calculates number of pairs of consecutive values in the data where the difference between is greater than 50.
    """

    @staticmethod
    def threshold():
        return 50


class RMSSD(TDFeature):
    """
    Calculates the square root of the mean of the squared differences.
    """

    def __init__(self, data=None, params=None):
        super(RMSSD, self).__init__(data, params)
        if len(data) < 2:
            print self.__class__.__name__ + " Warning: not enough samples (" + str(len(data)) + " < 2). " + \
                "To calculate the differences between consecutive values at least 2 samples are needed."
            self._value = np.nan
        else:
            diff = Diff.get(self._data)
            self._value = np.sqrt(sum(diff ** 2) / len(diff))


class SDSD(TDFeature):
    """Calculates the standard deviation of the differences between each value and its next."""

    def __init__(self, data=None, params=None):
        super(SDSD, self).__init__(data, params)
        diff = Diff.get(self._data)
        self._value = np.std(diff)


# TODO: fix documentation
class Triang(TDFeature):
    """Calculates the Triangular index."""

    def __init__(self, data=None, params=None):
        super(Triang, self).__init__(data, params)
        h, b = Histogram.get(self._data)
        self._value = len(self._data) / np.max(h)


# TODO: fix documentation
class TINN(TDFeature):
    """Calculates the difference between two histogram-related indexes."""

    def __init__(self, data=None, params=None):
        super(TINN, self).__init__(data, params)
        hist, bins = Histogram.get(self._data)
        max_x = HistogramMax.get(self._data)
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

        self._value = m - n
