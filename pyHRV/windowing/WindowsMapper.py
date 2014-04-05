__author__ = 'AleB'
__all__ = ['WindowsMapper']


class WindowsMapper(object):
    """Helps the iteration of an Index over a WindowsGenerator
    """

    def __init__(self, data, win_gen, index):
        self._data = data
        self._map = None
        self._wing = win_gen
        self._index = index

    def _comp_once(self, win):
        return self._index(data=self._data[win.begin, win.end]).value

    def compute(self):
        self._map = map(self._comp_once, self._wing)

    @property
    def results(self):
        return self._map
