# coding=utf-8
from __future__ import division
import filters.Filters
from indicators.TimeDomain import *
from indicators.FrequencyDomain import *
from indicators.NonLinearDomain import *
from indicators.PeaksDescription import *
from filters.Filters import *
from tools.Tools import *
from segmentation.SegmentsGenerators import *
import segmentation.SegmentsGenerators
from BaseSegmentation import Segment
from Signal import EvenlySignal, UnevenlySignal, SparseSignal

__author__ = "AleB"


def fmap(segments, algorithms, alt_signal=None):
    """
    Generates a list composed of a list of results for each segment.

    [[result for each algorithm] for each segment]
    :param segments: An iterable of segments (e.g. an initialized SegmentGenerator)
    :param algorithms: A list of algorithms
    :param alt_signal: The signal that will be used instead of the one referenced in the segments
    :return: A list containing a list for each segment containing a value for each algorithm
    """
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


