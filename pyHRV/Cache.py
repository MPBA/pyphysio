##ck2
# coding=utf-8
__author__ = 'AleB'

import numpy as np
from scipy import signal

from pyHRV.utility import build_takens_vector, interpolate_rr
from pyHRV.DataSeries import DataSeries
from pyHRV.PyHRVSettings import PyHRVDefaultSettings as Sett


class CacheableDataCalc(object):
    """
    Static class that calculates the data that can be cached.
    static = not instantiable, no instance methods or fields
    """

    def __init__(self):
        """
        @raise NotImplementedError: Ever, this class is static and not instantiable.
        """
        raise NotImplementedError(self.__class__.__name__ + " is static and not instantiable.")

    @classmethod
    def get(cls, data, params=None, use_cache=True):
        """
        Gets the data if cached or calculates it, saves it in the cache and returns it.
        @param data: Source data
        @type data: DataSeries
        @param params: Parameters for the calculator
        @param use_cache: Weather to use the cache memory or not
        @return: The final data
        @rtype: DataSeries
        """
        if use_cache and isinstance(data, DataSeries):
            if not data.cache_check(cls):
                data.cache_pre_calc_data(cls, params)
        else:
            return cls._calculate_data(data, params)
        return data.cache_get_data(cls)

    @classmethod
    def _calculate_data(cls, data, params):
        """
        Placeholder for the subclasses
        @raise NotImplementedError: Ever
        """
        raise NotImplementedError("Only on " + cls.__name__ + " sub-classes")

    @classmethod
    def cid(cls):
        """
        Gets an identifier for the class
        @rtype : str or unicode
        """
        return cls.__name__ + "_cn"


class FFTCalc(CacheableDataCalc):
    @classmethod
    def _calculate_data(cls, data, interp_freq):
        """
        Calculates the FFT data to cache
        @param data: DataSeries object
        @type data: DataSeries
        @param interp_freq: Frequency for the interpolation before the pow. spec. estimation.
        @return: Data to cache: (bands, powers)
        @rtype: (array, ndarray)
        """
        rr_interp, bt_interp = interpolate_rr(data.series, interp_freq)
        interp_freq = interp_freq
        hw = np.hamming(len(rr_interp))

        frame = rr_interp * hw
        frame = frame - np.mean(frame)

        spec_tmp = np.absolute(np.fft.fft(frame)) ** 2  # FFT
        powers = spec_tmp[0:(np.ceil(len(spec_tmp) / 2))]  # Only positive half of spectrum
        bands = np.linspace(start=0, stop=interp_freq / 2, num=len(powers), endpoint=True)  # frequencies vector
        return bands, powers


class PSDWelchCalc(CacheableDataCalc):
    @classmethod
    def _calculate_data(cls, data, interp_freq):
        """
        Calculates the PSDWelch data to cache
        @param data: DataSeries object
        @type data: DataSeries
        @param interp_freq: Frequency for the interpolation before the pow. spec. estimation.
        @return: Data to cache: (bands, powers, total_power)
        @rtype: (array, ndarray, float)
        """
        if interp_freq is None:
            interp_freq = Sett.default_interpolation_freq
        rr_interp, bt_interp = interpolate_rr(data, interp_freq)
        bands, powers = signal.welch(rr_interp, interp_freq, nfft=max(128, rr_interp.shape[-1]))
        powers = np.sqrt(powers)
        return bands, powers / np.max(powers), sum(powers) / len(powers)


class Histogram(CacheableDataCalc):
    @classmethod
    def _calculate_data(cls, data, histogram_bins=Sett.cache_histogram_bins):
        """
        Calculates the Histogram data to cache
        @param data: DataSeries object
        @type data: DataSeries
        @param histogram_bins: Histogram bins
        @return: Data to cache: (hist, bin_edges)
        @rtype: (array, array)
        """
        return np.histogram(data, histogram_bins)


class HistogramMax(CacheableDataCalc):
    @classmethod
    def _calculate_data(cls, data, histogram_bins=Sett.cache_histogram_bins):
        """
        Calculates the Histogram's max value
        @param data: DataSeries object
        @type data: DataSeries
        @param histogram_bins: Histogram bins
        @return: Data to cache: (hist, bin_edges)
        @rtype: (array, array)
        """
        h, b = Histogram.get(data, histogram_bins)
        return np.max(h)  # TODO: max h or b(max h)??


class RRDiff(CacheableDataCalc):
    @classmethod
    def _calculate_data(cls, data, params=None):
        """
        Calculates the differences between consecutive values
        @param data: DataSeries object
        @type data: DataSeries
        @param params: Unused
        @return: Data to cache: diff
        @rtype: array
        """
        return np.diff(np.array(data))


class StandardDeviation(CacheableDataCalc):
    @classmethod
    def _calculate_data(cls, data, params=None):
        """
        Calculates the standard deviation data
        @param data: DataSeries object
        @type data: DataSeries
        @param params: Unused
        @return: Data to cache: st. dev.
        @rtype: array
        """
        return np.std(np.array(data))


class BuildTakensVector2(CacheableDataCalc):
    @classmethod
    def _calculate_data(cls, data, params=None):
        """
        Calculates the the vector of the sequences of length 2 of the data
        @param data: DataSeries object
        @type data: DataSeries
        @param params: Unused
        @return: Data to cache: Takens vector (2)
        @rtype: array
        """
        return build_takens_vector(data, 2)


class BuildTakensVector3(CacheableDataCalc):
    @classmethod
    def _calculate_data(cls, data, params=None):
        """
        Calculates the the vector of the sequences of length 2 of the data
        @param data: DataSeries object
        @type data: DataSeries
        @param params: Unused
        @return: Data to cache: Takens vector (3)
        @rtype: array
        """
        return build_takens_vector(data, 3)


class PoinSD(CacheableDataCalc):
    @classmethod
    def _calculate_data(cls, data, params=None):
        """
        Calculates Poincaré SD 1 and 2
        @param data: DataSeries object
        @type data: DataSeries
        @param params: Unused
        @return: Data to cache: (SD1, SD2)
        @rtype: (array, array)
        """
        xd, yd = np.array(list(data[:-1])), np.array(list(data[1:]))
        sd1 = np.std((xd - yd) / np.sqrt(2.0))
        sd2 = np.std((xd + yd) / np.sqrt(2.0))
        return sd1, sd2
