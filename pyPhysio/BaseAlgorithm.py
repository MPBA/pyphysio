# coding=utf-8
from Signal import Signal
from abc import abstractmethod as _abstract, ABCMeta as _ABCMeta
from Utility import PhUI as _PhUI

__author__ = 'AleB'


class Algorithm(object):
    """
    This is the algorithm container super class. It (is abstract) should be used only to be extended.
    """
    __metaclass__ = _ABCMeta

    _params_descriptors = {}
    _parameter_error = None
    _logger = None

    def __init__(self, params=None, **kwargs):
        """
        Incorporates the parameters and saves them in the instance.
        @param params: Dictionary of string-value parameters passed by the user.
        @type params: dict
        @param _kwargs: Internal channel for subclasses kwargs parameters.
        @type _kwargs: dict
        @param kwargs: kwargs parameters to pass to the feature extractor.
        @type kwargs: dict
        """
        assert params is None or type(params) is dict, "The syntax is algorithm([params])(signal)"
        if params is None:
            self._params = {}
        else:
            self._params = params.copy()
        self._params.update(kwargs)
        # Parameters check
        p = self.get_params_descriptors()
        for n in p:
            if n in self._params:
                r, e = p[n](self._params, n)
                if not r:
                    self._parameter_error = ValueError("Error in parameters: " + e)
            else:
                r, self._params[n] = p[n].not_present(self._params, n, self)
                if not r:
                    self._parameter_error = ValueError("Error in parameters")

    def __call__(self, data):
        """
        Executes the algorithm using the parameters saved by the constructor.
        @param data: The data.
        @type data: TimeSeries
        @return: The result.
        """
        if self._parameter_error is None:
            return self.get(data, self._params)
        else:
            raise self._parameter_error

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
        if type(params) is dict:
            kwargs.update(params)
        if not isinstance(data, Signal):
            _PhUI.w("The data is not a Signal. Optimization is not available and some errors may be caused by missing meta data (e.g. sampling frequency).")
            use_cache = False
        if use_cache is True:
            Cache.cache_check(data)
            # noinspection PyTypeChecker
            return Cache.cache_get_data(data, cls, kwargs)
        else:
            return cls.algorithm(data, kwargs)

    def get_params(self):
        """
        Placeholder for the subclasses
        @return
        """
        return self._params

    @classmethod
    def get_params_descriptors(cls):
        """
        Returns the used parameters
        :rtype: dict[str, Parameter]
        """
        return cls._params_descriptors

    @classmethod
    @_abstract
    def is_compatible(cls, signal):
        """
        Placeholder for the subclasses
        :returns: Weather nature is compatible or not
        @raise NotImplementedError: Ever
        """
        pass

    @classmethod
    @_abstract
    def algorithm(cls, data, params):
        """
        Placeholder for the subclasses
        @raise NotImplementedError: Ever
        :param params:
        :param data:
        """
        pass

    @classmethod
    def cache_hash(cls, params):
        """
        This method computes an hash to use as a part of the key in the cache starting from the parameters used by the
        feature. Uses the method _utility_hash([par1,...parN])
        This class is abstract.
        @return: The hash of the parameters used by the feature.
        :param params:
        """
        p = params.copy()
        p.update({'': str(cls)})
        return cls._utility_hash(p)

    @staticmethod
    def _utility_hash(x):
        return str(x).replace('\'', '')

    @classmethod
    def log(cls, message):
        _PhUI.i(cls.__name__ + ": " + message)
        cls._logger((_PhUI.i, cls.__name__ + ": " + message))

    @classmethod
    def warn(cls, message):
        _PhUI.w(cls.__name__ + ": " + message)
        if cls._logger is not None:
            cls._logger((_PhUI.w, cls.__name__ + ": " + message))

    @classmethod
    def error(cls, message, raise_error=False):
        _PhUI.e(cls.__name__ + ": " + message)
        if raise_error:
            raise
        else:
            cls._logger((_PhUI.e, cls.__name__ + ": " + message))

    @classmethod
    def set_logger(cls, logger):
        cls._logger = logger

    @classmethod
    def unset_logger(cls):
        cls._logger = None

    @classmethod
    def emulate_log(cls, log):
        map(lambda (f, m): f(m), log)


class Cache(object):
    """ Class that gives a cache support."""

    def __init__(self):
        pass

    # Field-checked methods

    @staticmethod
    def cache_clear(self):
        """
        Clears the cache and frees memory (GC?)
        :param self:
        """
        self._cache = {}
        # setattr(self, "_cache", {})

    @staticmethod
    def cache_check(self):
        """
        Checks the presence of the cache structure.
        :param self:
        """
        if not hasattr(self, "_cache"):
            Cache.cache_clear(self)

    # Field-unchecked methods

    @staticmethod
    def cache_invalidate(self, calculator, params):
        """
        Invalidates the specified calculator's cached data if any.
        @type calculator: Algorithm
        """
        hh = calculator.cache_hash(params)
        if hh in self._cache:
            del self._cache[hh]

    @staticmethod
    def cache_get_data(self, calculator, params):
        """
        Gets data from the cache if valid
        @type calculator: Algorithm
        @return: The data or None
        """
        hh = calculator.cache_hash(params)
        log = []

        if hh not in self._cache:
            calculator.set_logger(lambda x: log.append(x))
            val = calculator.algorithm(self, params)
            self._cache[hh] = (val, log)
            calculator.unset_logger()
        else:
            val, log = self._cache[hh]
            calculator.emulate_log(log)
        return val

