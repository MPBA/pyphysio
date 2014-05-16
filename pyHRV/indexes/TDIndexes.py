__author__ = 'AleB'
__all__ = ['Mean', 'Median', 'SD', 'SDSD', 'NN10', 'NN25', 'NN50', 'NNx', 'PNN10', 'PNN25', 'PNN50', 'PNNx', 'RMSSD',
           'HRMean', 'HRMedian', 'HRSD', "Triang", "TINN"]

import numpy as np

from pyHRV.Cache import CacheableDataCalc, RRDiff, Histogram, HistogramMax
from pyHRV.indexes.BaseIndexes import TDIndex
from pyHRV.PyHRVSettings import PyHRVDefaultSettings as Sett


class Mean(TDIndex, CacheableDataCalc):
    """Calculates the average of the data series."""

    def __init__(self, data=None):
        super(Mean, self).__init__(data)
        self._value = Mean.get(self._data)

    @classmethod
    def _calculate_data(cls, data, params):
        return np.mean(data)

    @classmethod
    def calculate_on(cls, state):
        if state.ready():
            val = (state.sum() - state.last() + state.new()) / state.len()
        else:
            val = None
        return val


class HRMean(TDIndex, CacheableDataCalc):
    """Calculates the average of the data series and converts it into Beats per Minute."""

    def __init__(self, data=None):
        super(HRMean, self).__init__(data)
        self._value = HRMean.get(self._data)

    @classmethod
    def _calculate_data(cls, data, params):
        return np.mean(60000 / data)

    @classmethod
    def calculate_on(cls, state):
        if state.ready():
            val = 60000 / Mean.calculate_on(state)
        else:
            val = None
        return val


class Median(TDIndex, CacheableDataCalc):
    """Calculates the median of the data series."""

    def __init__(self, data=None):
        super(Median, self).__init__(data)
        self._value = Median.get(self._data)

    @classmethod
    def _calculate_data(cls, data, params):
        return np.median(data)


class HRMedian(TDIndex):
    """Calculates the average of the data series and converts it into Beats per Minute."""

    def __init__(self, data=None):
        super(HRMedian, self).__init__(data)
        self._value = 60000 / Median.get(self._data)


class SD(TDIndex, CacheableDataCalc):
    """Calculates the standard deviation of the data series."""

    def __init__(self, data=None):
        super(SD, self).__init__(data)
        self._value = SD.get(self._data)

    @classmethod
    def _calculate_data(cls, data, params):
        return np.std(data)


class HRSD(TDIndex, CacheableDataCalc):
    """Calculates the average of the data series and converts it into Beats per Minute."""

    def __init__(self, data=None):
        super(HRSD, self).__init__(data)
        self._value = HRSD.get(self._data)

    @classmethod
    def _calculate_data(cls, data, params):
        return np.std(60 / data)


class PNNx(TDIndex):
    """Calculates the presence proportion (0.0-1.0) in the data series of pairs of consecutive IBIs
    where the difference between the two values is greater than the default parameter."""

    def __init__(self, data=None, threshold=None):
        super(PNNx, self).__init__(data)
        self._xth = threshold if not threshold is None else self.threshold()
        self._value = NNx(data, threshold).value / float(len(data))

    @staticmethod
    def threshold():
        return Sett.nnx_default_threshold

    @classmethod
    def calculate_on(cls, state):
        NNx.calculate_on(state, cls.threshold()) / float(state.len)  # TODO: (AleB) Wrong


class NNx(TDIndex):
    """Calculates number of pairs of consecutive IBIs in the data series where the difference between
     the two values is greater than the default parameter."""

    def __init__(self, data=None, threshold=None):
        super(NNx, self).__init__(data)
        self._xth = threshold if not threshold is None else self.threshold()
        diff = RRDiff.get(self._data)
        self._value = sum(1 for x in diff if x > self._xth)

    @staticmethod
    def threshold():
        return Sett.nnx_default_threshold

    @classmethod
    def calculate_on(cls, state, threshold=None):  # TODO: (AleB) Wrong
        name = cls.__name__ + "_E4j2oj23"
        if threshold is None:
            threshold = cls.threshold()
        if state.ready() and state.len() >= 2:
            val = state.get(name)
            if not state.old() is None:
                if abs(state.old() - state.vec[0]) > threshold:
                    val -= 1
            if abs(state.vec[-1] - state.vec[-2]) > threshold:
                val += 1
            state.set(name, val)
        else:
            val = None
        return val


class PNN10(PNNx):
    """Calculates the presence proportion (0.0-1.0) in the data series of pairs of consecutive IBIs
    where the difference between the two values is greater than 10."""

    @staticmethod
    def threshold():
        return 10


class PNN25(PNNx):
    """Calculates the presence proportion (0.0-1.0) in the data series of pairs of consecutive IBIs
    where the difference between the two values is greater than 25."""

    @staticmethod
    def threshold():
        return 25


class PNN50(PNNx):
    """Calculates the presence proportion (0.0-1.0) in the data series of pairs of consecutive IBIs
    where the difference between the two values is greater than 50."""

    @staticmethod
    def threshold():
        return 50


class NN10(NNx):
    """Calculates number of pairs of consecutive IBIs in the data series where the difference between
     the two values is greater than 10."""

    @staticmethod
    def threshold():
        return 10


class NN25(NNx):
    """Calculates number of pairs of consecutive IBIs in the data series where the difference between
     the two values is greater than 25."""

    @staticmethod
    def threshold():
        return 25


class NN50(NNx):
    """Calculates number of pairs of consecutive IBIs in the data series where the difference between
     the two values is greater than 50."""

    @staticmethod
    def threshold():
        return 50


class RMSSD(TDIndex):
    """Calculates the ."""

    def __init__(self, data=None):
        super(RMSSD, self).__init__(data)
        diff = RRDiff.get(self._data)
        self._value = np.sqrt(sum(diff ** 2) / (len(diff)))


class SDSD(TDIndex):
    """Calculates the standard deviation of the differences between each value and its next."""

    def __init__(self, data=None):
        super(SDSD, self).__init__(data)
        diff = RRDiff.get(self._data)
        self._value = np.std(diff)


#TODO: fix docu
class Triang(TDIndex):
    """Calculates the Triangular HRV index."""

    def __init__(self, data=None):
        super(Triang, self).__init__(data)
        h, b = Histogram.get(self._data)
        self._value = len(self._data) / np.max(h)


#TODO: fix docu
class TINN(TDIndex):
    """Calculates the difference between two histogram-related indexes."""

    def __init__(self, data=None):
        super(TINN, self).__init__(data)
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
