__author__ = 'AleB'

from pandas import TimeSeries


class Feature(object):
    """
    This is the feature extractor super class. It should be used only to be extended.
    """

    def __init__(self, params=None, _kwargs=None, **kwargs):
        """
        Incorporates the parameters and saves them in the instance.
        @param params: Dictionary of string-value parameters passed by the user.
        @type params: dict
        @param _kwargs: Internal channel for subclasses kwargs parameters.
        @type _kwargs: dict
        @param kwargs: kwargs parameters to pass to the feature extractor.
        @type kwargs: dict
        @return: The callable instance.
        """
        assert self.__class__ != Feature, "This class is abstract and must be extended to be used."
        assert params is None or type(params) is dict
        assert type(_kwargs) is dict or _kwargs is None
        if params is None:
            self._params = {}
        else:
            self._params = params.copy()
        self._params.update(kwargs)
        if type(_kwargs) is dict:
            self._params.update(_kwargs)

    def __call__(self, data):
        """
        Computes the feature using the parameters saved by the constructor.
        @param data: The data where to extract the features.
        @type data: TimeSeries
        @return: The value of the feature.
        """
        return self.get(data, self._params)

    def __repr__(self):
        return self.__class__.__name__ + str(self._params)

    @classmethod
    def get(cls, data, params=None, use_cache=True, **kwargs):
        """
        Gets the data from the cache or calculates, caches and returns it.
        @param data: Source data
        @type data: TimeSeries
        @param params: Parameters for the calculator
        @type params: dict
        @param use_cache: Weather to use the cache memory or not
        @type use_cache: bool
        @return: The value of the feature.
        """
        assert type(data) is TimeSeries, "The data must be a pandas TimeSeries."
        if type(params) is dict:
            kwargs.update(params)
        if use_cache is True:
            Cache.cache_check(data)
            return Cache.cache_get_data(data, cls, kwargs)
        else:
            return cls.raw_compute(data, kwargs)

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
        feature. Uses the method _utility_hash([par1,...parN])
        This class is abstract.
        @return: The hash of the parameters used by the feature.
        """
        return cls._utility_hash([i + "=" + (str(params[i]) if i in params else '') for i in cls.get_used_params()] +
                                 [cls.__name__])

    @staticmethod
    def get_used_params():
        """
        Placeholder for the subclasses, if not overridden the feature should not use any parameter.
        """
        return []

    @staticmethod
    def _utility_hash(x):
        return str(x).__hash__()

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
    def cache_check(self):
        """
        Checks the presence of the cache structure.
        """
        if not hasattr(self, "_cache"):
            Cache.cache_clear(self)

    # Field-unchecked methods

    @staticmethod
    def cache_invalidate(self, calculator, params):
        """
        Invalidates the specified calculator's cached data if any.
        @type calculator: Feature
        """
        hh = calculator.cache_hash(params)
        if hh in self._cache:
            del self._cache[hh]

    @staticmethod
    def cache_get_data(self, calculator, params):
        """
        Gets data from the cache if valid
        @type calculator: Feature
        @return: The data or None
        """
        hh = calculator.cache_hash(params)
        if hh not in self._cache:
            self._cache[hh] = calculator.raw_compute(self, params)
        return self._cache[hh]


# noinspection PyAbstractClass
class TDFeature(Feature):
    """
    This is the base class for the Time Domain Indexes.
    """

    def __init__(self, params=None, _kwargs=None):
        super(TDFeature, self).__init__(params, _kwargs)


# noinspection PyAbstractClass
class FDFeature(Feature):
    """
    This is the base class for the Frequency Domain Features.
    It uses the interpolation frequency parameter interp_freq.
    """

    def __init__(self, params=None, _kwargs=None):
        super(FDFeature, self).__init__(params, _kwargs)
        assert 'interp_freq' in self._params, "This feature needs 'interp_freq'."
        self._interp_freq = self._params['interp_freq']

    @staticmethod
    def get_used_params():
        return ['interp_freq', 'psd_method']


# noinspection PyAbstractClass
class NonLinearFeature(Feature):
    """
    This is the base class for the Non Linear Features.
    """

    def __init__(self, params=None, _kwargs=None):
        super(NonLinearFeature, self).__init__(params, _kwargs)


# noinspection PyAbstractClass
class CacheOnlyFeature(Feature):
    """
    This is the base class for the generic features.
    """

    def __init__(self, params=None, _kwargs=None):
        super(CacheOnlyFeature, self).__init__(params, _kwargs)
