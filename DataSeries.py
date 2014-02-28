# Classes for cached data (RR) elaborations
#

import pandas as pd
import numpy as np
from utility import interpolate_rr


class DataSeries(pd.TimeSeries):
    """ Pandas' DataFrame class. Gives a cache support through CacheableDataCalc subclasses. """

    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False, metatag={}):
        """ Default constructor.
        @param data: Data to insert in the DataFrame
        @param columns: see Pandas doc
        @param dtype: see Pandas doc
        @param copy: see Pandas doc
        """
        self._cache = {}
        self.metatag = metatag
        super(DataSeries, self).__init__(data, index, columns, dtype, copy)

    def cache_clear(self):
        """ Clears the cache and frees memory (GC?)
        """
        self._cache = {}

    def cache_check(self, calculator):
        """ Check if the cache contains valid calculator's data
        :type calculator: CacheableDataCalc
        :param calculator: CacheableDataCalc
        :return: If the cache is valid
        """
        return calculator.cid() in self._cache

    def cache_invalidate(self, calculator):
        """
        :type calculator: CacheableDataCalc
        :param calculator: CacheableDataCalc
        """
        if self.cache_check(calculator):
            del self._cache[calculator.cid()]

    def cache_pre_calc_data(self, calculator, params):
        """ Precalculates data and caches it
        :type calculator: CacheableDataCalc
        :param calculator: CacheableDataCalc
        """
        # aggiungo alla cache
        self._cache[calculator.cid()] = calculator.get(self, params, use_cache=False)
        return self._cache[calculator.cid()]

    def cache_get_data(self, calculator):
        """ Gets data from the cache if valid
        :type calculator: CacheableDataCalc
        :param calculator: CacheableDataCalc subclass
        :return: The data or None
        """
        if self.cache_check(calculator):
            return self._cache[calculator.cid()]
        else:
            return None


class CacheableDataCalc(object):
    """ Static class that calculates cacheable data (like FFT etc.) """

    def __init__(self):
        raise TypeError('CacheableDataCalc is a static class')

    # metodo pubblico per ricavare i dati dalla cache o da _calculate_data(..)
    @classmethod
    def get(cls, data, params=None, use_cache=True):
        assert isinstance(data, DataSeries)
        if use_cache:
            if not data.cache_check(cls):
                data.cache_pre_calc_data(cls, params)
        else:
            return cls._calculate_data(data, params)
        return data.cache_get_data(cls)

    # metodo da sovrascrivere nelle sottoclassi
    @classmethod
    def _calculate_data(cls, data, params):
        raise NotImplementedError("Only on " + cls.__name__ + " sub-classes")

    # stringa usata come chiave nel dizionario cache
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
        assert isinstance(data, DataSeries)
        # calcolo FFT
        RR_interp, BT_interp = interpolate_rr(data.series, params)
        Finterp = params
        hw = np.hamming(len(RR_interp))

        frame = RR_interp * hw
        frame = frame - np.mean(frame)

        spec_tmp = np.absolute(np.fft.fft(frame)) ** 2  # calcolo FFT
        spec = spec_tmp[0:(np.ceil(len(spec_tmp) / 2))]  # Only positive half of spectrum
        freqs = np.linspace(start=0, stop=Finterp / 2, num=len(spec), endpoint=True)  # creo vettore delle frequenze
        return ((freqs, spec))


class RRDiff(CacheableDataCalc):
    @classmethod
    def _calculate_data(cls, data, params=None):
        """ Calculates the intermediate data
        :type data: DataSeries
        :param data: RRSeries object
        :param params: Params object
        :return: Data to cache
        """
        assert isinstance(data, DataSeries)
        return np.diff(data)

