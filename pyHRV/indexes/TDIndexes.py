from pyHRV.PyHRVSettings import PyHRVDefaultSettings

__all__ = ['HRMean', 'HRMedian', 'HRSTD', 'NNx', 'PNNx', 'RMSSD', 'Mean', 'Median', 'STD', 'SDSD']

import numpy as np

from pyHRV.Cache import CacheableDataCalc, RRDiff
from pyHRV.indexes.BaseIndexes import TDIndex


class Mean(TDIndex, CacheableDataCalc):
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
    def __init__(self, data=None):
        super(HRMean, self).__init__(data)
        self._value = HRMean.get(self._data)

    @classmethod
    def _calculate_data(cls, data, params):
        return np.mean(60 / data)


class Median(TDIndex, CacheableDataCalc):
    def __init__(self, data=None):
        super(Median, self).__init__(data)
        self._value = Median.get(self._data)

    @classmethod
    def _calculate_data(cls, data, params):
        return np.median(data)


class HRMedian(TDIndex):
    def __init__(self, data=None):
        super(HRMedian, self).__init__(data)
        self._value = 60 / Median.get(self._data)


class STD(TDIndex, CacheableDataCalc):
    def __init__(self, data=None):
        super(STD, self).__init__(data)
        self._value = STD.get(self._data)

    @classmethod
    def _calculate_data(cls, data, params):
        return np.std(data)


class HRSTD(TDIndex, CacheableDataCalc):
    def __init__(self, data=None):
        super(HRSTD, self).__init__(data)
        self._value = HRSTD.get(self._data)

    @classmethod
    def _calculate_data(cls, data, params):
        return np.std(60 / data)


class PNNx(TDIndex):
    def __init__(self, data=None, threshold=PyHRVDefaultSettings.TDIndexes.nnx_default_threshold):
        super(PNNx, self).__init__(data)
        self._xth = threshold
        self._value = NNx(data, threshold).value / len(data) / 100.0


class NNx(TDIndex):
    def __init__(self, data=None, threshold=None):
        super(NNx, self).__init__(data)
        if threshold is None:
            threshold = NNx.threshold()
        self._xth = threshold
        diff = RRDiff.get(self._data)
        self._value = 100.0 * sum(1 for x in diff if x > self._xth)

    @staticmethod
    def threshold():
        return PyHRVDefaultSettings.TDIndexes.nnx_default_threshold

    @classmethod
    def calculate_on(cls, state):
        name = cls.__name__ + "_E4j2oj23"
        if state.ready() and state.len() >= 2:
            val = state.get(name)
            if not state.old() is None:
                if abs(state.old() - state.vec[0]) > cls.threshold():
                    val -= 1
            if abs(state.vec[-1] - state.vec[-2]) > cls.threshold():
                val += 1
            state.set(name, val)
        else:
            val = None
        return val


class DGT10(NNx):
    @staticmethod
    def threshold():
        return 10


class DGT20(NNx):
    @staticmethod
    def threshold():
        return 20


class DGT50(NNx):
    @staticmethod
    def threshold():
        return 50


class RMSSD(TDIndex):
    def __init__(self, data=None):
        super(RMSSD, self).__init__(data)
        diff = RRDiff.get(self._data)
        self._value = np.sqrt(sum(diff ** 2) / (len(diff) - 1))


class SDSD(TDIndex):
    def __init__(self, data=None):
        super(SDSD, self).__init__(data)
        diff = RRDiff.get(self._data)
        self._value = np.std(diff)
