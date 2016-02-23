# coding=utf-8
from BaseSegmentation import SegmentationIterator
from PhUI import PhUI


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
        self._feats = indexes
        self._winn = -1
        self._params = params

    def __iter__(self):
        return SegmentationIterator(self)

    def _comp_one(self, win):
        ret = []
        win_ds = win(self._data)
        for algorithm in self._feats:
            if isinstance(algorithm, str) or isinstance(algorithm, unicode):
                assert False, "The string addressing is temporarily not supported"
                # p = getattr(dir(), algorithm) TODO not working
                # ret.append(p(win_ds))
            elif type(algorithm) is type:  # TODO bug: improve this check
                p = algorithm(self._params)
                ret.append(p(win_ds))
            else:
                PhUI.w("The specified algorithm '%s' is not an algorithm nor a PyPhysio algorithm name." % algorithm)
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
                PhUI.i("Processing " + str(w))
            self._map.append(self._comp_one(w))
        df = self._map
        # TODO test: is this functional?
        df.columns = self.labels()
        return df

    def labels(self):
        """
        Gets the labels of the table returned from the results property after the compute_all call.
        @rtype : list
        """
        ret = ['w_name', 'w_begin', 'w_end']
        for index in self._feats:
            if isinstance(index, str) | isinstance(index, unicode):
                assert False, "The string addressing is temporarily not supported"
                # index = getattr(pyPhysio, index)
            if isinstance(index, type):
                ret.append(index.__name__)
            else:
                ret.append(index.__repr__())
        return ret

    def results(self):
        """
        Returns the results table calculated in the compute_all call.
        @return: dict
        """
        return self._map
