# coding=utf-8
from abc import abstractmethod as _abstract, ABCMeta as _ABCMeta
from pyphysio.Signal import Signal, EvenlySignal
from pyphysio.Utility import PhUI as _PhUI
import numpy as _np
__author__ = 'AleB'


class Algorithm(object):
    """
    This is the algorithm container super class. It (is abstract) should be used only to be extended.
    """
    __metaclass__ = _ABCMeta

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
        self.set_unchecked(**kwargs)  # already checked by __init__

    def __call__(self, data):
        """
        Executes the algorithm using the parameters saved by the constructor.
        @param data: The data.
        @type data: TimeSeries
        @return: The result.
        """
        return self.run(data, self._params)

    def __repr__(self):
        return self.__class__.__name__ + str(self._params) if 'name' not in self._params else self._params['name']

    def set_unchecked(self, **kwargs):
        self._params.update(kwargs)

    def set(self, **kwargs):
        kk = self.get()
        kk.update(kwargs)
        self.__init__(**kk)

    def get(self, param=None):
        """
        Placeholder for the subclasses
        @return
        """
        if param is None:
            return self._params
        else:
            return self._params[param]

    @classmethod
    def run(cls, data, params=None, use_cache=False, **kwargs):
        """
        Gets the data from the cache or calculates, caches and returns it.
        @param data: Source data
        @type data: TimeSeries
        @param params: Parameters for the calculator
        @type params: dict
        @param use_cache: Whether to use the cache memory or not
        @type use_cache: bool
        @return: The value of the feature.
        """
        if type(params) is dict:
            kwargs.update(params)
        if not isinstance(data.get_values(), _np.ndarray):
            _PhUI.w("The data must be a Signal (see class EvenlySignal and UnevenlySignal).")
            use_cache = False
        if use_cache is True:
            Cache.cache_check(data)
            # noinspection PyTypeChecker
            return Cache.run_cached(data, cls, kwargs)
        else:            
            if not data.is_multi():
                return cls.algorithm(data, kwargs)
            else:
                data_values = data.get_values()
                values_out = []
                for i_ch in range(data.get_nchannels()):
                    channel_ph = EvenlySignal(data_values[:,i_ch], data.get_sampling_freq(), data.get_start_time())
                    output_ph = cls.algorithm(channel_ph, kwargs)
                    values_out.append(output_ph)
        
                # if output are signals, compose a multimodal instance
                if isinstance(values_out[0], EvenlySignal):
                    values_out_np = _np.stack([x.get_values() for x in values_out], axis=1)
                    output = data.clone_properties(values_out_np)
                    return(output)
                else:
                    return(values_out)

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
    def cache_key(cls, params):
        """
        This method computes an hash to use as a part of the key in the cache starting from the parameters used by the
        feature.
        @return: The hash of the parameters used by the feature.
        :param params:
        """
        p = params.copy()
        p.update({'': str(cls)})
        return str(p).replace('\'', '')

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
        map(lambda f_m: f_m[0](f_m[1]), log)


# noinspection PyProtectedMember
class Cache(object):
    """ Class that gives cache support."""

    def __init__(self):
        pass

    # Field-checked methods

    @staticmethod
    def cache_clear(obj):
        """
        Clears the cache and frees memory (GC?)
        :param obj:
        """
        obj._cache = {}
        obj._mutated = False

    @staticmethod
    def cache_check(obj):
        """
        Checks the presence of the cache structure.
        :param obj:
        """
        if not hasattr(obj, "_cache") or hasattr(obj, "_mutated") and obj._mutated:
            Cache.cache_clear(obj)

    # Field-unchecked methods

    @staticmethod
    def invalidate(obj, algorithm, params):
        """
        Invalidates the specified calculator's cached data if any.
        :type algorithm: Algorithm
        :param obj:
        :param params:
        """
        key = algorithm.cache_key(params)
        if key in obj._cache:
            del obj._cache[key]

    @staticmethod
    def run_cached(obj, algorithm, params):
        """
        Gets data from the cache if valid
        :param params:
        :param obj:
        :type algorithm: Algorithm
        :return: The data or None
        """
        key = algorithm.cache_key(params)

        if key not in obj._cache:
            algorithm.set_logger()
            val = algorithm.algorithm(obj, params)
            log = algorithm.unset_logger()
            obj._cache[key] = (val, log)
        else:
            val, log = obj._cache[key]
            algorithm.emulate_log(log)
        return val
