__author__ = 'AleB'
__all__ = ['NamedWinGen', 'LinearWinGen', 'CollectionWinGen']
from WindowsBase import WindowsGenerator, Window


class IBIWindow(Window):
    """Base IBI Window, a begin-end pair that provides the duration computation."""

    @property
    def duration(self):
        return sum(self._data[self._begin: self._end])

    def __repr__(self):
        return "Win(%d, %d, %s: %dms)" % (self.begin, self.end, self.name, self.duration)


class MixedWindow(Window):
    """Mixed states Window, a begin-center-end triad."""

    #TODO: AleB: Solve Nones problem as in Window (super)
    def __init__(self, begin=None, end=None, name=None, name2=None, center=None, copy=None):
        super(MixedWindow, self).__init__(begin, end, name, copy)
        if copy is None:
            self._name2 = name2
            self._center = center
        else:
            self._name2 = copy.name2
            self._center = copy.center

    @property
    def name2(self):
        return self._name2

    @property
    def center(self):
        return self._center

    def __repr__(self):
        return "MixedWin(%d, %s, %d, %s, %d)" % (self.begin, self.name, self.center, self.name2, self.end)


class LinearWinGen(WindowsGenerator):
    """Generates a linear set of windows (b+i*s, b+i*s+w)."""

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
    """Wraps a list of windows from an existing collection.
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


class NamedWinGen(WindowsGenerator):
    """Generates a list of windows from a labels list.
    """

    def __init__(self, data, labels, include_baseline_name=None):
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
