__author__ = 'AleB'

from pyHRV.DataSeries import Cache
from pyHRV.PyHRVSettings import MainSettings as Sett


class Feature(object):
    """
    This is an index extractor class.
    To calculate an index the relative class (subclass of this) must be instantiated,
    the resulting value will be available through the 'value' property. This class is abstract.
    """

    def __init__(self, data=None, value=None):
        """
        Initializes the index. This class is abstract.
        @param data: DataSeries from where extract the index.
        @type data: DataSeries
        @param value: Already present result.
        """
        assert self.__class__ != Feature
        self._value = value
        self._data = data

    @property
    def value(self):
        """
        Returns the value of the index, calculated during the instantiation.
        @rtype: float
        """
        return self._value

    @classmethod
    def required_sv(cls):
        """
        Returns the list of the support values that the computation of this index requires.
        @rtype: list
        """
        return []

    @classmethod
    def calculate_on(cls, state):
        """
        For on-line mode.
        @param state: Support values
        @raise NotImplementedError: Ever here.
        """
        raise TypeError(cls.__name__ + " is not available as an on-line index.")

    @classmethod
    def get(cls, data, params=None, use_cache=True):
        """
        Gets the data if cached or calculates it, saves it in the cache and returns it.
        @param data: Source data
        @param params: Parameters for the calculator
        @param use_cache: Weather to use the cache memory or not
        @return: The final data
        """
        if use_cache:
            if not Cache.cache_check(data, cls):
                Cache.cache_pre_calc_data(data, cls, params)
            return Cache.cache_get_data(data, cls)
        else:
            return cls._calculate_data(data, params)

    @classmethod
    def _calculate_data(cls, data, params):
        """
        Placeholder for the subclasses
        @raise NotImplementedError: Ever
        """
        raise NotImplementedError("Use a " + cls.__name__ + " sub-class")

    @classmethod
    def cid(cls):
        """
        Gets an identifier for the class
        @rtype : str or unicode
        """
        return cls.__name__ + "_cn"


class TDFeature(Feature):
    """
    This is the base class for the Time Domain Indexes.
    """

    @classmethod
    def _calculate_data(cls, data, params):
        raise TypeError("Use a " + cls.__name__ + " sub-class")

    def __init__(self, data=None):
        super(TDFeature, self).__init__(data)


class FDFeature(Feature):
    """
    This is the base class for the Frequency Domain Indexes.
    It uses the settings' default interpolation frequency parameter.
    """

    def __init__(self, interp_freq=Sett.default_interpolation_freq, data=None):
        super(FDFeature, self).__init__(data)
        self._interp_freq = interp_freq
        if len(data) < 3:
            raise TypeError("Not enough samples to perform a cube-spline interpolation.")


class NonLinearFeature(Feature):
    """
    This is the base class for the Non Linear Indexes.
    """

    def __init__(self, data=None):
        super(NonLinearFeature, self).__init__(data)


class CacheOnlyFeature(Feature):
    """
    This is the base class for the Non Linear Indexes.
    """

    def __init__(self, data=None):
        super(CacheOnlyFeature, self).__init__(data)
