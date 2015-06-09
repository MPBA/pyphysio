from __builtin__ import property

import pyPhysio
from pyPhysio.windowing.WindowsBase import WindowsGeneratorIterator
from pandas import DataFrame


__author__ = 'AleB'
__all__ = ['WindowsIterator']


class WindowsIterator(object):
    """
    Takes some features and calculates them on the given set of windows.
    Allows the iteration of the computation of a list of features over a WindowsGenerator.
    Use compute_all to execute the computation.
    """

    verbose = True

    def __init__(self, data, win_gen, indexes, params):
        """
        Initializes
        @param data: data on which compute windowed features
        @param win_gen: the windows generator
        @param indexes: list of classes as CLASS(DATA).value() ==> index value
        """
        self._data = data
        self._map = None
        self._wing = win_gen
        self._win_iter = win_gen.__iter__()
        self._index = indexes
        self._winn = -1
        self._params = params

    def __iter__(self):
        return WindowsGeneratorIterator(self)

    def _comp_one(self, win):
        ret = []
        win_ds = win.extract_data()
        for index in self._index:
            if isinstance(index, str) | isinstance(index, unicode):
                index = getattr(pyPhysio, index)
            ret.append(index(data=win_ds, params=self._params).value)
        self._winn += 1
        return [self._winn if win.label is None else win.label, win.begin, win.end] + ret

    def step_windowing(self):
        return self._comp_one(self._win_iter.next())

    def compute_all(self):
        """
        Executes the features computation (mapping with the windows).
        """
        self._map = []
        for w in self._wing:
            if WindowsIterator.verbose:
                print "Processing", w
            self._map.append(self._comp_one(w))
        df = DataFrame(self._map)
        df.columns = self.labels
        return df

    @property
    def labels(self):
        """
        Gets the labels of the table returned from the results property after the compute_all call.
        @rtype : list
        """
        ret = ['w_name', pyPhysio.MainSettings.load_windows_col_begin,
               pyPhysio.MainSettings.load_windows_col_end]
        for index in self._index:
            if isinstance(index, str) | isinstance(index, unicode):
                index = getattr(pyPhysio, index)
            ret.append(index.__name__)
        return ret

    @property
    def results(self):
        """
        Returns the results table calculated in the compute_all call.
        @return: dict
        """
        return self._map
