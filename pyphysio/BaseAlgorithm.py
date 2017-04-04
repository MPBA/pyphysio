# coding=utf-8
from pyphysio.Signal import Signal
from abc import abstractmethod as _abstract, ABCMeta as _ABCMeta
from pyphysio.Utility import PhUI as _PhUI

__author__ = 'AleB'


class Algorithm(object):
    """
    This is the algorithm container super class. It (is abstract) should be used only to be extended.
    """
    __metaclass__ = _ABCMeta

    _params_descriptors = {}
    _parameter_error = None
    _log = None

    def __init__(self, **kwargs):
        """
        Incorporates the parameters and saves them in the instance.
        @param params: Dictionary of string-value parameters passed by the user.
        @type params: dict
        @param _kwargs: Internal channel for subclasses kwargs parameters.
        @type _kwargs: dict
        @param kwargs: kwargs parameters to pass to the feature extractor.
        @type kwargs: dict
        """
        self._params = {}
        self.set(**kwargs)

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
        return self.__class__.__name__ + str(self._params) if 'name' not in self._params else self._params['name']

    def set(self, **kwargs):
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
            _PhUI.w("The data must be a Signal (see class EvenlySignal and UnevenlySignal).")
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
        l = (_PhUI.i, cls.__name__ + ": " + message)
        cls.emulate_log([l])
        if cls._log is not None:
            cls._log.append(l)

    @classmethod
    def warn(cls, message):
        l = (_PhUI.w, cls.__name__ + ": " + message)
        cls.emulate_log([l])
        if cls._log is not None:
            cls._log.append(l)

    @classmethod
    def error(cls, message, raise_error=False):
        l = (_PhUI.e, cls.__name__ + ": " + message)
        cls.emulate_log([l])
        if cls._log is not None:
            cls._log.append(l)
        assert not raise_error, l

    @classmethod
    def set_logger(cls):
        cls._log = []

    @classmethod
    def unset_logger(cls):
        u = cls._log
        cls._log = None
        return u

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
    def cache_invalidate(self, algorithm, params):
        """
        Invalidates the specified calculator's cached data if any.
        :type algorithm: Algorithm
        :param self:
        :param params:
        """
        hh = algorithm.cache_hash(params)
        if hh in self._cache:
            del self._cache[hh]

    @staticmethod
    def cache_get_data(self, calculator, params):
        """
        Gets data from the cache if valid
        :param params:
        :param self:
        :type calculator: Algorithm
        :return: The data or None
        """
        hh = calculator.cache_hash(params)

        if hh not in self._cache:
            calculator.set_logger()
            val = calculator.algorithm(self, params)
            log = calculator.unset_logger()
            self._cache[hh] = (val, log)
        else:
            val, log = self._cache[hh]
            calculator.emulate_log(log)
        return val

