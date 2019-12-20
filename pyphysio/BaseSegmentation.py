# coding=utf-8
from abc import abstractmethod as _abstract, ABCMeta as _ABCMeta
from copy import copy as _cpy
import numpy as _np
from .Utility import PhUI as _PhUI
from .BaseAlgorithm import Algorithm as _Algorithm, Cache
from .Signal import EvenlySignal as _EvenlySignal, UnevenlySignal as _UnevenlySignal

__author__ = 'AleB'


class Segment(object):
    """
    Base Segment, a time begin-end pair with a reference to the base signal and a name.
    """

    def __init__(self, begin, end, label=None, signal=None):
        """
        Creates a base Window
        @param begin: Begin sample index
        @param end: End sample index
        """
        self._begin = begin
        self._end = end
        self._label = label
        self._signal = signal

    def get_begin_time(self):
        return self._begin

    def get_end_time(self):
        return self._end

    def get_begin(self):
        return self._signal.get_idx(self.get_begin_time())

    def get_end(self):
        return self._signal.get_idx(self.get_end_time()) if self.get_end_time() is not None else None

    def get_duration(self):
        return (self.get_end_time() - self.get_begin_time()) if self.get_end_time() is not None else None

    def get_label(self):
        return self._label

    def is_empty(self):
        return self._signal is None or self.get_begin_time() >= len(self._signal.get_end_time())

    def __call__(self, data=None):
        if data is None:
            data = self._signal
        return data.segment_time(self.get_begin_time(), self.get_end_time())

    def __repr__(self):
        return '[%s:%s' % (str(self.get_begin_time()), str(self.get_end_time())) + (
            ":%s]" % self._label if self._label is not None else "]")


class SegmentsGenerator(_Algorithm):
    """
    Base and abstract class for the windows computation.
    """
    __metaclass__ = _ABCMeta

    @_abstract
    def __init__(self, **kwargs):
        super(SegmentsGenerator, self).__init__(**kwargs)
        self._signal = None

    @_abstract
    def next_segment(self):
        """
        Executes a segmentation step.
        @raise StopIteration: End of the iteration
        """
        raise StopIteration()

    @_abstract
    def init_segmentation(self):
        """
        Executes a segmentation step.
        @raise StopIteration: End of the iteration
        """
        raise NotImplementedError()

    # Algorithm Override, no cache
    def __call__(self, data=None):
        assert data is not None or self._signal is not None, "No signal specified for " + self.__class__.__name__
        return self.run(data if data is not None else self._signal, self._params, use_cache=False)

    def __iter__(self):
        return SegmentationIterator(self)

    @classmethod
    def algorithm(cls, data, params):
        o = cls(**params)
        o._signal = data
        return o

    @classmethod
    def is_nature_supported(cls, data):
        return isinstance(data, _EvenlySignal)

    @classmethod
    def get_used_params(cls):
        return []

    def __repr__(self):
        if self._signal is not None:
            return super(SegmentsGenerator, self).__repr__() + " over\n" + str(self._signal)
        else:
            return super(SegmentsGenerator, self).__repr__()

    #REWRITE the run method for compatibility with MultiEvenly signals
    #Segments should not work on each channel
    #But output a sehments with multiple channels
    #to be processed by the algorithms
    
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
            return cls.algorithm(data, kwargs)            
                    
class SegmentationIterator(object):
    """
    A generic iterator that is called from each WindowGenerator from the __iter__ method.
    """

    def __init__(self, win):
        assert isinstance(win, SegmentsGenerator)
        self._win = _cpy(win)
        self._win.init_segmentation()

    def __next__(self):
        return self._win.next_segment()

    # Python 2 & users compatibility
    def next(self):
        return self.__next__()


class SegmentationError(Exception):
    """
    Generic Segmentation error.
    """
    pass
