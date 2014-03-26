# coding=utf-8
__author__ = "AleB"

__all__ = ['DataSeries']

import pandas as pd
from numpy import mean as npmean
from pyHRV.PyHRVSettings import PyHRVDefaultSettings as Sett


class DataSeries(pd.TimeSeries):
    """ Pandas' DataFrame class. Gives a cache support through CacheableDataCalc subclasses. """

    def __init__(self, data=None, copy=False, meta_tag=None):
        """ Default constructor.
        @param data: Data to insert in the DataFrame
        @param copy: see Pandas doc
        """
        super(DataSeries, self).__init__(data=data, copy=copy)
        self._cache = {}
        if meta_tag is None:
            self.meta_tag = {}
        else:
            self.meta_tag = meta_tag
        mean = npmean(data)
        assert (not Sett.TimeUnitCheck.time_unit_check_ibi_mean_max < Sett.TimeUnitCheck.time_unit_check_ibi
                | (mean < Sett.TimeUnitCheck.time_unit_check_ibi_mean_min)),\
            Sett.TimeUnitCheck.time_unit_check_ibi_warn % mean

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
        """ Pre-calculates data and caches it
        :type calculator: CacheableDataCalc
        :param calculator: CacheableDataCalc
        """
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
