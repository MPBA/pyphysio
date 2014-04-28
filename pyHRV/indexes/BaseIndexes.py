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


class SupportValue(object):
    """Abstract class that defines the SupportValues' interface
    """

    def enqueuing(self, new_value):
        """Updates the support-value with the new enqueued value.
        """
        pass

    def dequeuing(self, old_value):
        """Updates the support-value with the just dequeued value.
        """
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
