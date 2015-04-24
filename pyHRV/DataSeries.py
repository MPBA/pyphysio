__author__ = "AleB"
__all__ = ['Cache', 'DataSeries']

from pyHRV.indexes.BaseFeatures import Feature
from pandas import Series


class DataSeries(Series):
    def _box_item_values(self, key, values):
        raise NotImplementedError()

    def _constructor_sliced(self):
        raise NotImplementedError()


class Cache():
    """ Class that gives a cache support. Uses Feature."""

    def __init__(self):
        pass

    # Field-checked methods

    @staticmethod
    def cache_clear(self):
        """
        Clears the cache and frees memory (GC?)
        """
        setattr(self, "_cache", {})

    @staticmethod
    def cache_check(self, calculator, params):
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
            return calculator.cache_hash(params) in self._cache

    # Field-unchecked methods

    @staticmethod
    def cache_invalidate(self, calculator, params):
        """
        Invalidates the specified calculator's cached data if any.
        @param calculator: Cacheable data calculator
        @type calculator: CacheableDataCalc
        """
        if self.cache_check(calculator, params):
            del self._cache[calculator.cache_hash(params)]

    @staticmethod
    def cache_comp_and_save(self, calculator, params):
        """
        Calculates data and caches it
        @param calculator: Cacheable data calculator
        @type calculator: CacheableDataCalc
        """
        assert isinstance(calculator, Feature)
        self._cache[calculator.cache_hash(params)] = calculator.get(self, params, use_cache=False)
        return self._cache[calculator.cache_hash(params)]

    @staticmethod
    def cache_get_data(self, calculator, params):
        """
        Gets data from the cache if valid
        @param calculator: Cacheable data calculator
        @type calculator: CacheableDataCalc
        @return: The data or None
        @rtype: DataSeries or None
        """
        if self.cache_check(calculator, params):
            return self._cache[calculator.cache_hash(params)]
        else:
            return None
