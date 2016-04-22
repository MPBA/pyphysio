# coding=utf-8
from __future__ import division
import indicators.Indicators
import filters.Filters
from indicators.Indicators import *
from filters.Filters import *
from tools.Tools import *
from segmentation.SegmentsGenerators import *
import segmentation.SegmentsGenerators
from BaseSegmentation import Segment
from WindowsIterator import WindowsIterator
from Signal import EvenlySignal, UnevenlySignal, SparseSignal

__author__ = "AleB"


def fmap(segments, algorithms, alt_signal=None):
    return [[ind(seg(alt_signal)) for ind in algorithms] for seg in segments]


def algo(function, params=None):
    """
    Builds on the fly a new algorithm class using the passed function and params if passed.
    Note that the cache will not be available.
    :param function: function(data, params) to be called
    :param params: parameters to pass to the function.
    :return: An algorithm class if params is None else a parametrized algorithm instance.
    """

    from BaseAlgorithm import Algorithm

    class Custom(Algorithm):
        # def __call__(self, data):
        #     """
        #     Executes the algorithm using the parameters saved by the constructor.
        #     @param data: The data.
        #     @type data: TimeSeries
        #     @return: The result.
        #     """
        #     return self.get(data, self._params, use_cache=False)

        @classmethod
        def algorithm(cls, data, params):
            return function(data, params)

    if params is None:
        return Custom
    else:
        return Custom(params)


class PhUI(object):
    @staticmethod
    def a(condition, message):
        if not condition:
            raise ValueError(message)

    @staticmethod
    def o(mex):
        PhUI.p(mex, '', 31)

    @staticmethod
    def i(mex):
        PhUI.p(mex, '', 35)

    @staticmethod
    def w(mex):
        PhUI.p(mex, 'Warning: ', 33)

    @staticmethod
    def e(mex):
        PhUI.p(mex, 'Error: ', 34)

    @staticmethod
    def p(mex, lev, col):
        print(">%s\x1b[%dm%s\x1b[39m" % (lev, col, mex))