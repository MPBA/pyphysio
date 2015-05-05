__author__ = 'AleB'

from pandas import TimeSeries


class Feature(object):
    """
    This is the feature extractor super class.
    To calculate a feature the relative class (subclass of this) must be instantiated,
    the resulting value will be available through the 'value' property. This class is abstract.
    """

    def __init__(self, params=None, _kwargs=None, **kwargs):
        """
        Initializes the index. This class is abstract.
        @param data: DataSeries from where extract the index.
        @type data: DataSeries
        """
        assert self.__class__ != Feature, "Class is abstract."
        if type(params) is not dict:
            self._params = kwargs
        else:
            self._params = params.copy()
        self._params.update(kwargs)
        if type(_kwargs) is dict:
            self._params.update(_kwargs)

    def __call__(self, data):
        return self.__class__.get(data, self._params)

    @classmethod
    def get(cls, data, params=None, use_cache=True, **kwargs):
        """
        Gets the data if cached or calculates it, saves it in the cache and returns it.
        @param data: Source data
        @param params: Parameters for the calculator
        @param use_cache: Weather to use the cache memory or not
        @return: The final data
        """
        assert type(data) is TimeSeries, "The data must be a pandas TimeSeries."
        assert type(use_cache) is bool, "Need a boolean here."
        if params is None:
            params = kwargs
        else:
            params.update(kwargs)
        if use_cache:
            if not Cache.cache_check(data, cls, params):
                Cache.cache_comp_and_save(data, cls, params)
            return Cache.cache_get_data(data, cls, params)
        else:
            return cls.raw_compute(data, params)

    @classmethod
    def raw_compute(cls, data, params):
        """
        Placeholder for the subclasses
        @raise NotImplementedError: Ever
        """
        raise NotImplementedError(cls.__name__ + " is not implemented.")

    @classmethod
    def cache_hash(cls, params):
        """
        This method gives an hash to use as a part of the key in the cache starting from the parameters used by the
        feature. The method _utility_hash([par1,...parN])
        This class is abstract.
        @return: The hash of the parameters used by the cache feature.
        """
        return cls._utility_hash([params[i] for i in cls.get_used_params() if i in params] +
                                 [cls.__name__, "_cn"])

    @staticmethod
    def get_used_params():
        """
        Placeholder for the subclasses
        """
        return []

    @staticmethod
    def _utility_hash(x):
        assert isinstance(x, list), "Need a list of values, not a " + str(type(x))
        concatenation = "this is random salt "  # this is random salt
        for y in x:
            concatenation += str(y)
        concatenation += " adding bias"
        return concatenation.__hash__() % (2 ** 32)

    @classmethod
    def compute_on(cls, state):
        """
        For on-line mode.
        @param state: Support values
        @raise NotImplementedError: Ever here.
        """
        raise TypeError(cls.__name__ + " is not available as an on-line feature.")

    @classmethod
    def required_sv(cls):
        """
        Returns the list of the support values that the computation of this index requires.
        @rtype: list
        """
        return []


class Cache(object):
    """ Class that gives a cache support. Uses Feature."""

    def __init__(self):
        pass

    # Field-checked methods

    @staticmethod
    def cache_clear(self):
        """
        Clears the cache and frees memory (GC?)
        """
        setattr(self, "_cache", {})

    @staticmethod
    def cache_check(self, calculator, params):
        """
        Checks the presence in the cache of the specified calculator's data.
        @param calculator: Cacheable data calculator
        @return: Presence in the cache
        @rtype: Boolean
        """
        if not hasattr(self, "_cache"):
            setattr(self, "_cache", {})
            return False
        else:
            return calculator.cache_hash(params) in self._cache

    # Field-unchecked methods

    @staticmethod
    def cache_invalidate(self, calculator, params):
        """
        Invalidates the specified calculator's cached data if any.
        @param calculator: Cacheable data calculator
        """
        if self.cache_check(calculator, params):
            del self._cache[calculator.cache_hash(params)]

    @staticmethod
    def cache_comp_and_save(self, calculator, params):
        """
        Calculates data and caches it
        @param calculator: Cacheable data calculator
        """
        h = calculator.cache_hash(params)
        self._cache[h] = calculator.get(self, params, use_cache=False)
        return self._cache[h]

    @staticmethod
    def cache_get_data(self, calculator, params):
        """
        Gets data from the cache if valid
        @param calculator: Cacheable data calculator
        @type calculator: CacheableDataCalc
        @return: The data or None
        @rtype: DataSeries or None
        """
        if Cache.cache_check(self, calculator, params):
            return self._cache[calculator.cache_hash(params)]
        else:
            return None


# noinspection PyAbstractClass
class TDFeature(Feature):
    """
    This is the base class for the Time Domain Indexes.
    """

    def __init__(self, data=None, params=None):
        super(TDFeature, self).__init__(data, params)


# noinspection PyAbstractClass
class FDFeature(Feature):
    """
    This is the base class for the Frequency Domain Indexes.
    It uses the settings' default interpolation frequency parameter.
    """

    def __init__(self, data=None, params=None):
        super(FDFeature, self).__init__(data, params)
        self._interp_freq = params['interp_freq'] if params is not None and 'interp_freq' in params else 4
        if len(data) < 3:
            raise TypeError("Not enough samples to perform a cube-spline interpolation.")

    @staticmethod
    def get_used_params():
        return ['interp_freq', 'psd_method']


# noinspection PyAbstractClass
class NonLinearFeature(Feature):
    """
    This is the base class for the Non Linear Indexes.
    """

    def __init__(self, data=None, params=None):
        super(NonLinearFeature, self).__init__(data, params)


# noinspection PyAbstractClass
class CacheOnlyFeature(Feature):
    """
    This is the base class for the Non Linear Indexes.
    """

    def __init__(self, data=None, params=None):
        super(CacheOnlyFeature, self).__init__(data, params)
