__author__ = 'AleB'
__all__ = ['WindowError', 'Window', 'WindowsGenerator']
from pandas import Series
from copy import copy as cpy


class WindowError(Exception):
    """
    Generic Windowing error.
    """
    pass


class Window(object):
    """
    Base Window, a begin-end pair.
    """

    def __init__(self, begin, end, data=None):
        """
        Creates a base Window
        @param begin: Begin sample/time index
        @param end: End sample/time index
        """
        self._begin = begin
        self._end = end
        self._data = data

    @property
    def begin(self):
        return self._begin

    @property
    def end(self):
        return self._end

    @property
    def samples(self):
        return self._end - self._begin

    @property
    def duration(self):
        if self._data is None:
            return None
        else:
            return self._data.iget(self._end) - self._data.iget(self._begin)

    @property
    def label(self):
        return None

    def slice_data(self):
        return Series(self._data[self._begin: self._end])

    def __repr__(self):
        return '%d:%d' % (self.begin, self.end)


class WindowsGeneratorIterator(object):
    """
    A generic iterator that is called from each WindowGenerator from the __iter__ method.
    """

    def __init__(self, win):
        assert isinstance(win, WindowsGenerator)
        self._win = cpy(win)
        self._win.init_windowing()

    def next(self):
        return self._win.step_windowing()


class WindowsGenerator(object):
    """
    Base and abstract class for the windows computation.
    """

    def __iter__(self):
        return WindowsGeneratorIterator(self)

    def init_windowing(self):
        """
        Initializes the windowing generator
        """
        raise NotImplementedError()

    def step_windowing(self):
        """
        Executes a windowing step.
        @raise StopIteration: End of the iteration
        """
        raise StopIteration
