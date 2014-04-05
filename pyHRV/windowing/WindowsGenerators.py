__author__ = 'AleB'
__all__ = ['LinearWinGen']
from WindowsBase import WindowsGenerator, Window


class LinearWinGen(WindowsGenerator):
    """Generates a linear set of windows (b+i*s, b+i*s+w)
    """

    def __init__(self, begin, step, width, data=None, end=None):
        super(LinearWinGen, self).__init__(data)
        if data is None and end is None:
            raise ValueError("Don't know where to find the length: data or end parameter must be not None.")
        self._begin = begin
        if end is None:
            self._end = len(data)
        else:
            self._end = end
        self._step = step
        self._width = width
        self._begin = begin
        self._pos = self._begin

    def step_windowing(self):
        b, e = (self._pos, self._pos + self._width)
        if e > self._end:
            self._pos = self._begin
            raise StopIteration
        else:
            self._pos += self._step
            return Window(b, e)


class CollectionWinGen(WindowsGenerator):
    """Wraps a set of windows from an existing collection
    """

    def __init__(self, data, win_list):
        super(CollectionWinGen, self).__init__(data)
        self._wins = win_list
        self._ind = 0

    def step_windowing(self):
        if self._ind >= len(self._wins):
            raise StopIteration
        else:
            self._ind += 1
            assert self._wins[self._ind] is Window
            return self._wins[self._ind]
