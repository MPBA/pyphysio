# coding=utf-8
from __future__ import division
import indicators.Indicators
import filters.Filters
from indicators.Indicators import *
from filters.Filters import *
from BaseAlgorithm import CustomAlgorithm as _CustomAlgorithm
from segmentation.SegmentsGenerators import *
import segmentation.SegmentsGenerators
from BaseSegmentation import Segment
from WindowsIterator import WindowsIterator
from Signal import *

__author__ = "AleB"


def fmap(segments, algorithms, alt_signal=None):
    return [[ind(seg(alt_signal)) for ind in algorithms] for seg in segments]


def algo(algorithm, check_params=None, check_signal_type=None):
    return _CustomAlgorithm(algorithm, check_params, check_signal_type)
