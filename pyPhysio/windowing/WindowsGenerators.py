__author__ = 'AleB'
__all__ = ['NamedWinGen', 'LinearWinGen', 'LinearTimeWindows', 'CollectionWinGen']
import numpy as np

from WindowsBase import WindowsGenerator, Window
from pandas import Series


class TimeWindower(WindowsGenerator):
    """
    Base class
    """

    def __init__(self, data):
        """
        Initializes the win generator
        @param data: Data (mandatory) for the time computations
        """
        super(TimeWindower, self).__init__()
        self._i = 0
        self._t = 0
        self._ei = len(data)
        self._data = data

    def _next_sample(self, plus):
        tt = self._t
        ii = self._i
        t = self._t + plus
        while tt < t and ii < self._ei:
            tt += self._data[ii]
            ii += 1
        return tt, ii


class TimeWindows(TimeWindower):
    def __init__(self, step, width, data=None, length=None, begin_time=0):
        assert data is not None or length is not None, "Either data or length must be not None"
        assert data is None or isinstance(data, Series)
        super(TimeWindows, self).__init__(data)
        if length is None:
            self._end = len(data)
        else:
            self._end = length
        self._begin = begin_time
        self._step = step
        self._width = width
        self._pos = self._begin

    def init_windowing(self):
        self._pos = self._begin

    def step_windowing(self):
        b, e = (self._pos, self._pos + self._width)
        if e > self._end:
            raise StopIteration
        else:
            self._pos += self._step
            return Window(b, e, self._data)


class LinearTimeWindows(TimeWindows):
    """
    Generates a linear-timed set of Time windows (b+i*s, b+i*s+w).
    """

    def __init__(self, step, width, data, begin=None, end=None):
        """
        Initializes the win generator
        @param step: Step samples
        @param width: Width of the window
        @param data: Data of the windows point
        @param end: End index or None for the end of the data specified
        """
        super(LinearTimeWindows, self).__init__(data)
        self._step_t = step
        self._width_t = width
        self._begin = begin
        self._end = end
        self._bt, self._bi = self._t, self._i
        if self._begin is not None:
            self._bt, self._bi = self._next_sample(self._begin)
        if self._end is not None:
            et = np.sum(self._data)
            while self._end < et:
                self._ei -= 1
                et -= self._data[self._ei - 1]
        self._init()

    def _init(self):
        self._t, self._i = self._bt, self._bi

    def step_windowing(self):
        if self._i < self._ei:
            tt, ii = self._next_sample(self._width_t)
            w = IBIWindow(self._i, ii, self._data)
            self._t, self._i = self._next_sample(self._step_t)
            return w
        else:
            self._init()
            raise StopIteration


class CollectionWinGen(WindowsGenerator):
    """
    Wraps a list of windows from an existing collection.
    """

    def __init__(self, win_list, data=None):
        """
        Initializes the win generator
        @param win_list: List of Windows to consider
        @type win_list: Sized Iterable
        @param data: Data of the windows point
        """
        super(CollectionWinGen, self).__init__(data)

        self._wins = win_list
        self._ind = 0

    def step_windowing(self):
        if self._ind >= len(self._wins):
            self._ind = 0
            raise StopIteration
        else:
            self._ind += 1
            assert isinstance(self._wins[self._ind - 1], Window)
            return self._wins[self._ind - 1]


class NamedWinGen(WindowsGenerator):
    """
    Generates a list of windows from a labels list.
    """

    def __init__(self, data, include_baseline_name=None):
        """
        Initializes the win generator
        @param data: Data to window
        """
        super(NamedWinGen, self).__init__(data)
        self._i = -1
        self._ibn = include_baseline_name
        if not self._data.has_labels():
            raise TypeError("Data has no labels.")
        l, self._is, t = (None, None, None)  # self._data.get_labels()

    def step_windowing(self):
        self._i += 1
        if self._i < len(self._is) - 1:
            return IBIWindow(self._is[self._i], self._is[self._i + 1], self._data)
        elif self._i < len(self._is):
            return IBIWindow(self._is[self._i], len(self._data), self._data)
        else:
            self._i = -1
            raise StopIteration()

    def __repr__(self):
        return "<%s - labels: %d at 0x%hx>" % (
            self.__class__.__name__, len(self._is), id(self))
