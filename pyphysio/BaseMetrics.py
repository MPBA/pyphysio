'''
THIS CODE IS UNDER DEVELOPMENT
'''

# coding=utf-8
from pyphysio.BaseAlgorithm import Algorithm
from abc import ABCMeta as _ABCMeta
from .Signal import Signal
from .Utility import PhUI as _PhUI
__author__ = 'AleB'


class Metrics(Algorithm):
    """
    Algorithms that take as input two signals and return a scalar value indicating the similarity/distance
    """
    __metaclass__ = _ABCMeta
    
    
    @classmethod
    def run(cls, data1, data2, params=None, use_cache=False, **kwargs):
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
        if not isinstance(data1, Signal) or not isinstance(data2, Signal):
            _PhUI.w("The data must be a Signal (see class EvenlySignal and UnevenlySignal).")
        return cls.algorithm(data1, data2, kwargs)
