# coding=utf-8
__author__ = 'AleB'


class DataAnalysis(object):
    pass


class Index(object):
    def __init__(self, data=None, value=None):
        self._value = value
        self._data = data

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

    def update(self, data):
        self._data = data
        self._value = None


class TDIndex(Index):
    def __init__(self, data=None, value=None):
        super(TDIndex, self).__init__(data, value)


class FDIndex(Index):
    def __init__(self, interp_freq, data=None, value=None):
        super(FDIndex, self).__init__(data, value)
        self._interp_freq = interp_freq


class NonLinearIndex(Index):
    def __init__(self, data=None, value=None):
        super(NonLinearIndex, self).__init__(data, value)


class RRFilters(DataAnalysis):
    """ Static class containing methods for filtering RR intervals data. """

    def __init__(self):
        raise NotImplementedError("RRFilters is a static class")

    @staticmethod
    def example_filter(series):
        """ Example filter method, does nothing
        :param series: DataSeries object to filter
        :return: DataSeries object filtered
        """
        ### assert type(series) is DataSeries
        return series

        # xTODO: add analysis scripts like in the example
