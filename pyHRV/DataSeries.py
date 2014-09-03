__author__ = "AleB"
__all__ = ['DataSeries', 'data_series_from_ecg', 'data_series_from_bvp']

from numpy import min, max, mean, diff, array

import pandas as pd

from pyHRV.PyHRVSettings import MainSettings as Sett
from pyHRV.Utility import peak_detection


def data_series_from_bvp(bvp, bvp_time, delta_ratio=Sett.import_bvp_delta_max_min_numerator,
                         filters=Sett.import_bvp_filters):
    """
    Loads an IBI (RR) data series from a BVP data set and filters it with the specified filters list.
    @param delta_ratio: delta parameter for the peak detection
    @type delta_ratio: float
    @param bvp: ecg values column
    @type bvp: Iterable
    @param bvp_time: ecg timestamps column
    @type bvp_time: Iterable
    @param filters: sequence of filters to be applied to the data (e.g. from IBIFilters)
    @return: Filtered signal DataSeries
    @rtype: DataSeries
    """
    delta = (max(bvp) - min(bvp)) / delta_ratio
    max_i, ii, iii, iv = peak_detection(bvp, delta, bvp_time)
    s = DataSeries(diff(max_i) * 1000)
    for f in filters:
        s = f(s)
    s.meta_tag['from_type'] = "data_time-bvp"
    s.meta_tag['from_peak_delta'] = delta
    s.meta_tag['from_freq'] = mean(diff(bvp_time))
    s.meta_tag['from_filters'] = list(Sett.import_bvp_filters)
    return s


def data_series_from_ecg(ecg, ecg_time, delta=Sett.import_ecg_delta, filters=Sett.import_bvp_filters):
    """
    Loads an IBI (RR) data series from an ECG data set and filters it with the specified filters list.
    @param delta: delta parameter for the peak detection
    @type delta: float
    @param ecg: ecg values column
    @type ecg: Iterable
    @param ecg_time: ecg timestamps column
    @type ecg_time: Iterable
    @return: Filtered signal DataSeries
    @rtype: DataSeries
    """
    # TODO: explain delta
    max_tab, min_tab, ii, iii = peak_detection(ecg, delta, ecg_time)
    s = DataSeries(diff(max_tab))
    for f in filters:
        s = f(s)
    s.meta_tag['from_type'] = "data_time-ecg"
    s.meta_tag['from_peak_delta'] = delta
    s.meta_tag['from_freq'] = mean(diff(ecg_time))
    s.meta_tag['from_filters'] = list(Sett.import_ecg_filters)
    return s


def derive_holdings(data, labels):
    ll = []
    tt = []
    ii = []
    ts = 0
    pre = None
    for i in xrange(len(labels)):
        if pre != labels[i]:
            ll.append(labels[i])
            tt.append(ts)
            ii.append(i)
            ts += data[i]
            pre = labels[i]
    return ll, tt, ii


class DataSeries(pd.Series):
    """ Pandas' Series class extension that gives a cache support through CacheableDataCalc's subclasses."""

    def __init__(self, data=None, copy=False, meta_tag=None, labels=None, labels_event_times=None):
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
        if not labels is None:
            if labels_event_times is None and len(labels) == len(data):
                self._labels, self._timestamps, self._samples = derive_holdings(data, labels)
            elif not labels_event_times is None and len(data) >= len(labels) == len(labels_event_times):
                self._labels = array(labels)
                self._timestamps = array(labels_event_times)
            else:
                raise ValueError("Labels format not valid.")
        else:
            self._labels = self._timestamps = self._samples = None

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

    def get_labels(self):
        """
        Returns a tuple composed by the label names their start timestamps and their start indexes
        @return:
        """
        return self._labels, self._samples, self._timestamps

    def has_labels(self):
        return not self._labels is None
