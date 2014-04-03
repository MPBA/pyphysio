# coding=utf-8
from pyHRV.PyHRVSettings import PyHRVDefaultSettings

__all__ = ['HRMean', 'HRMedian', 'HRSTD', 'NNx', 'PNNx', 'RMSSD', 'RRMean', 'RRMedian', 'RRSTD', 'SDSD']

import numpy as np

from pyHRV.Cache import CacheableDataCalc, RRDiff
from pyHRV.indexes.BaseIndexes import TDIndex


class RRMean(TDIndex, CacheableDataCalc):
    def __init__(self, data=None):
        super(RRMean, self).__init__(data)
        self._value = RRMean.get(self._data)

    @classmethod
    def _calculate_data(cls, data, params):
        return np.mean(data)

    @classmethod
    def calculate_on(cls, state):
        if state.ready():
            val = (state.sum() - state.old() + state.new()) / state.len()
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


class RRMedian(TDIndex, CacheableDataCalc):
    def __init__(self, data=None):
        super(RRMedian, self).__init__(data)
        self._value = RRMedian.get(self._data)

    @classmethod
    def _calculate_data(cls, data, params):
        return np.median(data)


class HRMedian(TDIndex):
    def __init__(self, data=None):
        super(HRMedian, self).__init__(data)
        self._value = 60 / RRMedian.get(self._data)


class RRSTD(TDIndex, CacheableDataCalc):
    def __init__(self, data=None):
        super(RRSTD, self).__init__(data)
        self._value = RRSTD.get(self._data)

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


## self._value= NNx/len(diff) >> not convenient for a parameter problem
class PNNx(TDIndex):
    def __init__(self, data=None, threshold=PyHRVDefaultSettings.TDIndexes.nnx_default_threshold):
        super(PNNx, self).__init__(data)
        self._xth = threshold
        self._value = NNx(data, threshold).value / len(data)


class NNx(TDIndex):
    def __init__(self, data=None, threshold=PyHRVDefaultSettings.TDIndexes.nnx_default_threshold):
        super(NNx, self).__init__(data)
        self._xth = threshold
        diff = RRDiff.get(self._data)
        self._value = 100.0 * sum(1 for x in diff if x > self._xth)


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
