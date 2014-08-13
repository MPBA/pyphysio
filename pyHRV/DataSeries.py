##ck3

__author__ = "AleB"
__all__ = ['DataSeries']

import pandas as pd


class DataSeries(pd.Series):
    """ Pandas' Series class extension that gives a cache support through CacheableDataCalc's subclasses."""

    def __init__(self, data=None, copy=False, meta_tag=None):
        """
        Constructor
        @param data: Data to be stored.
        @param copy: If to copy the data (see Pandas doc.)
        @param meta_tag: Dict that stores meta-data tags about the data.
        """
        super(DataSeries, self).__init__(data=data, copy=copy)
        self._cache = {}
        if meta_tag is None:
            self.meta_tag = {}
        else:
            self.meta_tag = meta_tag

    def cache_clear(self):
        """
        Clears the cache and frees memory (GC?)
        """
        self._cache = {}

    def cache_check(self, calculator):
        """
        Checks the presence in the cache of the specified calculator's data.
        @param calculator: Cacheable data calculator
        @type calculator: CacheableDataCalc
        @return: Presence in the cache
        @rtype: Boolean
        """
        return calculator.cid() in self._cache

    def cache_invalidate(self, calculator):
        """
        Invalidates the specified calculator's cached data if any.
        @param calculator: Cacheable data calculator
        @type calculator: CacheableDataCalc
        """
        if self.cache_check(calculator):
            del self._cache[calculator.cid()]

    def cache_pre_calc_data(self, calculator, params):
        """
        Calculates data and caches it
        @param calculator: Cacheable data calculator
        @type calculator: CacheableDataCalc
        """
        self._cache[calculator.cid()] = calculator.get(self, params, use_cache=False)
        return self._cache[calculator.cid()]

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

    def __getitem__(self, win):
        return DataSeries(self[win.begin: win.end], True, self.meta_tag)
