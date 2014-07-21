##ck3
__author__ = 'AleB'
__all__ = ['NamedWinGen', 'LinearWinGen', 'LinearTimeWinGen', 'CollectionWinGen']
import numpy as np

from WindowsBase import WindowsGenerator, Window


class IBIWindow(Window):
    """Base IBI Window, a begin-end pair that provides the duration computation."""

    def __init__(self, begin, end, data, name=None):
        """
        Creates a time Window
        @param begin: Begin sample index
        @param end: End sample index
        @param data: IBI data from where to calculate the duration
        @param name: Label for the window
        """
        Window.__init__(self, begin, end, name)
        self._data = data

    @property
    def duration(self):
        """
        Time duration of the window (sum of IBIs)
        @rtype: float
        """
        return sum(self._data[self._begin: self._end])

    def __repr__(self):
        return "Win(%d, %d, %s: %dms)" % (self.begin, self.end, self.name, self.duration)


class MixedWindow(Window):
    """
    Mixed states Window, a begin-center-end triad.
    """

    def __init__(self, begin, end, center, name=None, name2=None):
        """
        Creates a mixed labels Window
        @param begin: Begin sample index
        @param end: End sample index
        @param name: First label for the window
        @param name2: Second label for the window
        """
        super(MixedWindow, self).__init__(begin, end, name)
        self._name2 = name2
        self._center = center

    @property
    def name2(self):
        return self._name2

    @property
    def center(self):
        return self._center

    def __repr__(self):
        return "MixedWin(%d, %s, %d, %s, %d)" % (self.begin, self.name, self.center, self.name2, self.end)


class LinearWinGen(WindowsGenerator):
    """
    Generates a linear-index set of windows (b+i*s, b+i*s+w).
    """

    def __init__(self, begin, step, width, data=None, end=None):
        """
        Initializes the win generator
        @param begin: Start index
        @param step: Step samples
        @param width: Width of the window
        @param data: Data of the windows point
        @param end: End index or None for the end of the data specified
        @raise ValueError: When no data and no end are specified
        """
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


class LinearTimeWinGen(WindowsGenerator):
    """
    Generates a linear-timed set of Time windows (b+i*s, b+i*s+w).
    """

    def __init__(self, step, width, data, end=None):
        """
        Initializes the win generator
        @param step: Step samples
        @param width: Width of the window
        @param data: Data of the windows point
        @param end: End index or None for the end of the data specified
        """
        super(LinearTimeWinGen, self).__init__(data)
        self._step = step * 1000
        self._width = width * 1000
        self._cums = [0]
        self._cums.extend(np.cumsum(data))
        self._pos = 0
        if end is None:
            self._end = len(data) - 1  # RR sum should be the total time
        else:
            self._end = end

    def step_windowing(self):
        b = e = self._pos
        if b < self._end:
            et = self._cums[self._pos] + self._width
            while e <= self._end and et > self._cums[e]:
                e += 1
            if e > self._end:
                self._pos = 0
                raise StopIteration
            else:
                nt = self._cums[self._pos] + self._step
                while e < self._end and nt > self._cums[self._pos]:
                    self._pos += 1
                return Window(b, e)
        else:
            raise StopIteration


class CollectionWinGen(WindowsGenerator):
    """
    Wraps a list of windows from an existing collection.
    """

    def __init__(self, win_list, data=None):
        """
        Initializes the win generator
        @param win_list: List of Windows to consider list(Window)
        @param data: Data of the windows point
        """
        super(CollectionWinGen, self).__init__(data)
        self._wins = win_list
        self._ind = 0

    def step_windowing(self):
        if self._ind >= len(self._wins):
            raise StopIteration
        else:
            self._ind += 1
            assert isinstance(self._wins[self._ind - 1], Window)
            return self._wins[self._ind - 1]


class NamedWinGen(WindowsGenerator):
    """
    Generates a list of windows from a labels list.
    """

    def __init__(self, data, labels, include_baseline_name=None):
        """
        Initializes the win generator
        @param data: Data to window
        @param labels: List of the labels (one per sample) to consider
        @type labels: list(str) or list(unicode)
        """
        super(NamedWinGen, self).__init__(data)
        self._l = labels
        self._s = 0
        self._i = 0
        self._ibn = include_baseline_name

    def step_windowing(self):
        if self._i >= len(self._l):
            raise StopIteration()
        while self._i < len(self._l) and (self._l[self._s] == self._l[self._i] or self._ibn == self._l[self._i]):
            self._i += 1
        w = Window(self._s, self._i - 1, self._l[self._s])
        self._s = self._i
        return w

    def __repr__(self):
        return "<%s - labels: %d, step: %s object at 0x%hx>" % (
            self.__class__.__name__, len(self._l), self._s, id(self))
