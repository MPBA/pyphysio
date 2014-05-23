__author__ = 'AleB'

import numpy as np
from scipy import signal

from pyHRV.utility import build_takens_vector, interpolate_rr
from pyHRV.DataSeries import DataSeries
from pyHRV.PyHRVSettings import PyHRVDefaultSettings as Sett


class CacheableDataCalc(object):
    """ Static class that calculates cacheable data (like FFT etc.) """

    def __init__(self):
        pass

    @classmethod
    def get(cls, data, params=None, use_cache=True):
        if use_cache and isinstance(data, DataSeries):
            if not data.cache_check(cls):
                data.cache_pre_calc_data(cls, params)
        else:
            return cls._calculate_data(data, params)
        return data.cache_get_data(cls)

    @classmethod
    def _calculate_data(cls, data, params):
        raise NotImplementedError("Only on " + cls.__name__ + " sub-classes")

    @classmethod
    def cid(cls):
        """ Gets an identifier for the class
        :rtype : str
        """
        return cls.__name__ + "_cn"


class FFTCalc(CacheableDataCalc):
    @classmethod
    def _calculate_data(cls, data, params=None):
        """ Calculates the intermediate data
        :type data: DataSeries
        :param data: RRSeries object
        :param params: Params object
        :return: Data to cache
        """
        rr_interp, bt_interp = interpolate_rr(data.series, params)
        interp_freq = params
        hw = np.hamming(len(rr_interp))

        frame = rr_interp * hw
        frame = frame - np.mean(frame)

        spec_tmp = np.absolute(np.fft.fft(frame)) ** 2  # FFT
        powers = spec_tmp[0:(np.ceil(len(spec_tmp) / 2))]  # Only positive half of spectrum
        bands = np.linspace(start=0, stop=interp_freq / 2, num=len(powers), endpoint=True)  # frequencies vector
        return bands, powers


class PSDWelchCalc(CacheableDataCalc):
    @classmethod
    def _calculate_data(cls, data, to_freq):
        """ Calculates the intermediate data
        :type data: DataSeries
        :param data: RRSeries object
        :param to_freq: Sampling frequency
        :return: Data to cache
        """
        if to_freq is None:
            to_freq = Sett.default_interpolation_freq
        rr_interp, bt_interp = interpolate_rr(data, to_freq)
        bands, powers = signal.periodogram(rr_interp, to_freq, nfft=max(128, rr_interp.shape[-1]))
        powers = np.sqrt(powers)
        return bands, powers / np.max(powers), sum(powers) / len(powers)


class Histogram(CacheableDataCalc):
    @classmethod
    def _calculate_data(cls, data, params=None):
        """ Calculates the intermediate data
        :type data: DataSeries
        :param data: RRSeries object
        :param params: Params object
        :return: Data to cache
        """
        return np.histogram(data, Sett.cache_histogram_bins)


class HistogramMax(CacheableDataCalc):
    @classmethod
    def _calculate_data(cls, data, params=None):
        h, b = Histogram.get(data)
        return np.max(h)


class RRDiff(CacheableDataCalc):
    @classmethod
    def _calculate_data(cls, data, params=None):
        """ Calculates the intermediate data
        :type data: DataSeries
        :param data: RRSeries object
        :param params: Params object
        :return: Data to cache
        """
        return np.diff(np.array(data))


class StandardDeviation(CacheableDataCalc):
    @classmethod
    def _calculate_data(cls, data, params=None):
        """ Calculates the intermediate data
        :type data: DataSeries
        :param data: RRSeries object
        :param params: Params object
        :return: Data to cache
        """
        return np.std(np.array(data))


class BuildTakensVector2(CacheableDataCalc):
    @classmethod
    def _calculate_data(cls, data, params=None):
        return build_takens_vector(data, 2)


class BuildTakensVector3(CacheableDataCalc):
    @classmethod
    def _calculate_data(cls, data, params=None):
        return build_takens_vector(data, 3)


class PoinSD(CacheableDataCalc):
    @classmethod
    def _calculate_data(cls, data, params=None):
        xd, yd = np.array(list(data[:-1])), np.array(list(data[1:]))
        sd1 = np.std((xd - yd) / np.sqrt(2.0))
        sd2 = np.std((xd + yd) / np.sqrt(2.0))
        return sd1, sd2
