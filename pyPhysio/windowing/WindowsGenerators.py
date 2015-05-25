__author__ = 'AleB'
__all__ = ['TimeWindows', 'LabeledWindows', 'CollectionWindows']

from WindowsBase import WindowsGenerator, Window


class TimeWindows(WindowsGenerator):
    def __init__(self, step, width=0, start=0):
        super(TimeWindows, self).__init__()
        self._step = step
        self._width = step if width == 0 else width
        self._i = start

    def step_windowing(self):
        o = self._i
        self._i += self._step
        return Window(o, o + self._width, '')


class CollectionWindows(WindowsGenerator):
    """
    Wraps a list of windows from an existing collection.
    """

    def __init__(self, win_list):
        """
        Initializes the win generator
        @param win_list: List of Windows to consider
        """
        super(CollectionWindows, self).__init__()

        self._wins = win_list
        self._ind = 0

    def step_windowing(self):
        if self._ind >= len(self._wins):
            self._ind = 0
            raise StopIteration
        else:
            self._ind += 1
            assert isinstance(self._wins[self._ind - 1], Window), "%d is not a Window" % self._wins[self._ind - 1]
            return self._wins[self._ind - 1]


class LabeledWindows(WindowsGenerator):
    """
    Generates a list of windows from a labels list.
    """

    def __init__(self, labels, include_baseline_name=None):
        """
        Initializes the win generator
        @param labels: Labels time series
        """
        super(LabeledWindows, self).__init__()
        self._i = 0
        self._ibn = include_baseline_name
        self._labels = labels

    def step_windowing(self):
        if self._i < len(self._labels) - 1:
            w = Window(self._labels.index[self._i], self._labels.index[self._i + 1], self._labels[self._i])
        elif self._i < len(self._labels):
            from datetime import datetime as dt, MAXYEAR
            w = Window(self._labels.index[self._i], dt(MAXYEAR, 12, 31, 23, 59, 59, 999999999), self._labels[self._i])
        else:
            raise StopIteration()
        return w
