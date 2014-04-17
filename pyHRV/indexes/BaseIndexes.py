__author__ = 'AleB'


class DataAnalysis(object):
    pass


class Index(object):
    def __init__(self, data=None, value=None):
        self._value = value
        self._data = data

    @property
    def value(self):
        return self._value

    # on-line part
    @classmethod
    def calculate_on(cls, state):
        raise NotImplementedError(cls.__name__ + " is not available as an on-line index.")


class TDIndex(Index):
    def __init__(self, data=None):
        super(TDIndex, self).__init__(data)


class FDIndex(Index):
    def __init__(self, interp_freq, data=None):
        super(FDIndex, self).__init__(data)
        self._interp_freq = interp_freq


class NonLinearIndex(Index):
    def __init__(self, data=None):
        super(NonLinearIndex, self).__init__(data)


class RRFilters(DataAnalysis):
    """ Static class containing methods for filtering RR intervals data. """

    @staticmethod
    def example_filter(series):
        """ Example filter method, does nothing
        :param series: DataSeries object to filter
        :return: DataSeries object filtered
        """
        ### assert type(series) is DataSeries
        return series

        # xTODO: add filtering scripts like in the example


class SupportValues(object):
    def __init__(self, window=50):
        self._last = []
        self._win_size = window
        self._p = {}
        self._sum = 0
        self._len = 0
        self._old = None
        self._balance = None
        self._max = None

    def get(self, index, default=0):
        if not index in self._p:
            self._p[index] = default
        return self._p[index]

    def set(self, index, value):
        self._p[index] = value

    @property
    def old(self):
        return self._last[0]

    @property
    def last(self):
        return self._last[0]

    @property
    def new(self):
        return self._last[-1]

    @property
    def vec(self):
        return self._last

    def update(self, values):
        for a in values:
            self._enqueue(a)
        if self._win_size >= 0:
            while self.len > self._win_size:
                self._dequeue()

    @property
    def len(self):
        return self._len

    @property
    def sum(self):
        return self._sum

    @property
    def min(self):
        return self._sum

    @property
    def max(self):
        return self._sum

    @property
    def ready(self):
        return self._win_size < 0 < self.len or self.len == self._win_size

    def _enqueue(self, val):
        self._last.append(val)
        self._sum += val

    def _dequeue(self):
        val = self._last[0]
        del self._last[0]
        self._sum -= val
        self._old = val
        self._len -= 1
