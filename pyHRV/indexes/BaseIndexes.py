__author__ = 'AleB'


class SupportValues(object):
    def __init__(self, window=-1):
        self._last = []
        self._win_size = window
        self._p = {}
        self._sum = 0
        self._len = 0
        self._old = None

    def get(self, index, default=0):
        if not index in self._p:
            self._p[index] = default
        return self._p[index]

    def set(self, index, value):
        self._p[index] = value

    def old(self):
        return self._last[0]

    def last(self):
        return self._last[0]

    def new(self):
        return self._last[-1]

    @property
    def vec(self):
        return self._last

    def update(self, values):
        for a in values:
            self._enqueue(a)
        if self._win_size >= 0:
            while self.len() > self._win_size:
                self._dequeue()

    def len(self):
        return self._len

    def sum(self):
        return self._sum

    def ready(self):
        return self.len() == self._win_size

    def _enqueue(self, val):
        self._last.append(val)
        self._sum += val
        self._len += 1

    def _dequeue(self):
        self._sum -= self._last[0]
        self._old = self._last[0]
        del self._last[0]
        self._len -= 1


class DataAnalysis(object):
    pass


class Index(object):
    def __init__(self, data=None, value=None):
        self._value = value
        self._data = data

    def compute_block(self, data=None):
        pass

    # Offline part
    @property
    def calculated(self):
        """
        Returns weather the index is already calculated and up-to-date
        @return: Boolean
        """
        return not (self._value is None)

    @property
    def value(self):
        return self._value

    # on-line part
    @classmethod
    def update(cls, state):
        raise NotImplementedError(cls.__name__ + " is not available as an on-line index.")

    # Windowing part
    def compute_on_windows(self):
        pass


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
