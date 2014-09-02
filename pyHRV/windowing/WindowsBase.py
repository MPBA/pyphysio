from pyHRV.DataSeries import DataSeries

__author__ = 'AleB'
__all__ = ['WindowError', 'Window', 'WindowsGenerator']


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
        @param begin: Begin sample index
        @param end: End sample index
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
    def len(self):
        return self._end - self._begin

    def extract_data(self):
        return DataSeries(self._data[self._begin: self._end])

    def __repr__(self):
        return '%d:%d' % (self.begin, self.end)


class WindowsIterator(object):
    """
    A generic iterator that is called from each WindowGenerator from the __iter__ method.
    """

    def __init__(self, win):
        assert isinstance(win, WindowsGenerator)
        self._win = win

    def next(self):
        return self._win.step_windowing()


class WindowsGenerator(object):
    """
    Base and abstract class for the windows computation.
    """

    def __init__(self, data=None):
        self._data = None
        if data is None:
            pass
        else:
            self.init_windowing(data)
            self._winn = 0

    def __iter__(self):
        return WindowsIterator(self)

    def init_windowing(self, data):
        """
        Initializes the windowing generator
        @param data: Data to window
        """
        self._data = data

    def step_windowing(self):
        """
        Executes a windowing step.
        @raise StopIteration: End of the iteration
        """
        raise StopIteration
