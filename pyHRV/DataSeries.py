__author__ = "AleB"
__all__ = ['DataSeries', 'data_series_from_ecg', 'data_series_from_bvp']

import pandas as pd
from pyHRV.indexes.BaseFeatures import Feature


class Cache():
    """ Class extension that gives a cache support through CacheableDataCalc's subclasses."""

    def __init__(self):
        pass

    # Checked methods

    @staticmethod
    def cache_clear(self):
        """
        Clears the cache and frees memory (GC?)
        """
        setattr(self, "_cache", {})

    @staticmethod
    def cache_check(self, calculator):
        """
        Checks the presence in the cache of the specified calculator's data.
        @param calculator: Cacheable data calculator
        @type calculator: CacheableDataCalc
        @return: Presence in the cache
        @rtype: Boolean
        """
        assert isinstance(calculator, Feature)
        if not hasattr(self, "_cache"):
            setattr(self, "_cache", {})
            return False
        else:
            return calculator.cid() in self._cache

    # Safe methods

    @staticmethod
    def cache_invalidate(self, calculator):
        """
        Invalidates the specified calculator's cached data if any.
        @param calculator: Cacheable data calculator
        @type calculator: CacheableDataCalc
        """
        if self.cache_check(calculator):
            del self._cache[calculator.cid()]

    @staticmethod
    def cache_pre_calc_data(self, calculator, params):
        """
        Calculates data and caches it
        @param calculator: Cacheable data calculator
        @type calculator: CacheableDataCalc
        """
        self._cache[calculator.cid()] = calculator.get(self, params, use_cache=False)
        return self._cache[calculator.cid()]

    @staticmethod
    def cache_get_data(self, calculator):
        """
        Gets data from the cache if valid
        @param calculator: Cacheable data calculator
        @type calculator: CacheableDataCalc
        @return: The data or None
        @rtype: DataSeries or None
        """
        if self.cache_check(calculator):
            return self._cache[calculator.cid()]
        else:
            return None


class DataSeries(pd.Series, Cache):
    def _box_item_values(self, key, values):
        raise NotImplementedError()

    def _constructor_sliced(self):
        raise NotImplementedError()
