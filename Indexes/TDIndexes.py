# coding=utf-8

from DataSeries import CacheableDataCalc, RRDiff
from Indexes import TDIndex
import numpy as np


class RRMean(TDIndex, CacheableDataCalc):
    def __init__(self, data=None):
        super(RRMean, self).__init__(data)
        self._value = RRMean.get(self._data)

    @classmethod
    def _calculate_data(cls, data, params):
        return np.mean(data)


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
        super(TDIndex, self).__init__(data)
        self._value = HRSTD.get(self._data)

    @classmethod
    def _calculate_data(cls, data, params):
        return np.std(60 / data)


## self._value= NNx/len(diff) >> not convenient for a parameter problem
class PNNx(TDIndex):
    def __init__(self, threshold, data=None):
        super(TDIndex, self).__init__(data)
        self._xth = threshold
        self._value = NNx(threshold, data).value / len(data)


class NNx(TDIndex):
    def __init__(self, threshold, data=None):
        super(TDIndex, self).__init__(data)
        self._xth = threshold
        diff = RRDiff.get(self._data)
        self._value = 100.0 * sum(1 for x in diff if x > self._xth)


class RMSSD(TDIndex):
    def __init__(self, data=None):
        super(TDIndex, self).__init__(data)
        diff = RRDiff.get(self._data)
        self._value = np.sqrt(sum(diff ** 2) / (len(diff) - 1))


class SDSD(TDIndex):
    def __init__(self, data=None):
        super(TDIndex, self).__init__(data)
        diff = RRDiff.get(self._data)
        self._value = np.std(diff)
