from __builtin__ import property

import pyHRV
from pyHRV.windowing.WindowsBase import WindowsIterator
from pyHRV.DataSeries import DataSeries

__author__ = 'AleB'
__all__ = ['WindowsMapper']


class WindowsMapper(object):
    """Takes some indexes and calculates them on the given set of windows.
    Allows the iteration of the computation of a list of indexes over a WindowsGenerator.
    Use compute_all to execute the computation."""

    def __init__(self, data, win_gen, indexes):
        """Initializes

        @param data: data on which compute windowed indexes
        @param win_gen: the windows generator
        @param indexes: list of classes as CLASS(DATA).value() ==> index value
        """
        self._data = data
        self._map = None
        self._wing = win_gen
        self._win_iter = win_gen.__iter__()
        self._index = indexes

    def __iter__(self):
        return WindowsIterator(self)

    def _comp_one(self, win):
        ret = []
        win_ds = DataSeries(self._data[win.begin: win.end])
        for index in self._index:
            if isinstance(index, str) | isinstance(index, unicode):
                index = getattr(pyHRV, index)
            ret.append(index(data=win_ds).value)
        return [win.name, win.begin, win.end] + ret

    def step_windowing(self):
        return self._comp_one(self._win_iter.next())

    def compute_all(self):
        """Executes the indexes computation (mapping with the windows)."""
        self._map = map(self._comp_one, self._wing)

    @property
    def labels(self):
        """Gets the labels of the table returned from the results property after the compute_all call.
        @rtype : list
        """
        ret = ['w_name', pyHRV.PyHRVDefaultSettings.load_windows_col_begin,
               pyHRV.PyHRVDefaultSettings.load_windows_col_end]
        for index in self._index:
            if isinstance(index, str) | isinstance(index, unicode):
                index = getattr(pyHRV, index)
            ret.append(index.__name__)
        return ret

    @property
    def results(self):
        """
        Returns the results table calculated in the compute_all call.
        @return: dict
        """
        return self._map
