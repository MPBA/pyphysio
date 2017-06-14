# coding=utf-8
from __future__ import division

from numpy import array as _array

from .filters import Filters
from .segmentation import SegmentsGenerators
from .indicators import FrequencyDomain
from .indicators import NonLinearDomain
from .indicators import PeaksDescription
from .indicators import TimeDomain
from .BaseSegmentation import Segment
from .Signal import EvenlySignal, UnevenlySignal, from_pickle, from_pickleable
from .interactive import Annotate
# BE CAREFUL with NAMES!!!
from .estimators.Estimators import *
from .filters.Filters import *
from .indicators.FrequencyDomain import *
from .indicators.NonLinearDomain import *
from .indicators.PeaksDescription import *
from .indicators.TimeDomain import *
from .tests import TestData
from .segmentation.SegmentsGenerators import *
from .tools.Tools import *

__author__ = "AleB"


def preset_hrv_fd(prefix="IBI_"):
    VLF = PowerInBand(interp_freq=4, freq_max=0.04, freq_min=0.00001, method='ar', name="VLF_Pow")
    LF = PowerInBand(interp_freq=4, freq_max=0.15, freq_min=0.04, method='ar', name="LF_Pow")
    HF = PowerInBand(interp_freq=4, freq_max=0.4, freq_min=0.15, method='ar', name="HF_Pow")
    Total = PowerInBand(interp_freq=4, freq_max=2, freq_min=0.00001, method='ar', name="Total_Pow")
    
    t = [VLF, LF, HF, Total]

    if prefix is not None:
        for i in t:
            i.set(name=prefix + i.get("name"))

    return t


def preset_hrv_td(prefix="IBI_"):
    rmssd = RMSSD(name="RMSSD")
    sdsd = SDSD(name="SDSD")
    RRmean = Mean(name="Mean")
    RRstd = StDev(name="RRstd")
    RRmedian = Median(name="Median")
    pnn10 = PNNx(threshold=10, name="pnn10")
    pnn25 = PNNx(threshold=25, name="pnn25")
    pnn50 = PNNx(threshold=50, name="pnn50")
    mn = Min(name="Min")
    mx = Max(name="Max")
    sd1 = PoincareSD1(name="sd1")
    sd2 = PoincareSD2(name="sd2")
    sd12 = PoincareSD1SD2(name="sd12")
    sdell = PoinEll(name="sdell")
    DFA1 = DFAShortTerm(name="DFA1")
    DFA2 = DFALongTerm(name="DFA2")

    t = [rmssd, sdsd, RRmean, RRstd, RRmedian, pnn10, pnn25, pnn50, mn, mx, sd1, sd2, sd12,
         sdell, DFA1, DFA2]

    if prefix is not None:
        for i in t:
            i.set(name=prefix + i.get("name"))

    return t


def preset_phasic(delta, prefix="pha_"):
    mean = Mean()
    std = StDev()
    rng = Range()
    pks_max = PeaksMax(delta=delta)
    pks_min = PeaksMin(delta=delta)
    pks_mean = PeaksMean(delta=delta)
    n_peaks = PeaksNum(delta=delta)
    dur_mean = DurationMean(delta=delta, win_pre=2, win_post=2)
    slopes_mean = SlopeMean(delta=delta, win_pre=2, win_post=2)
    auc = AUC()

    t = [mean, std, rng, pks_max, pks_min, pks_mean, n_peaks, dur_mean, slopes_mean, auc]

    if prefix is not None:
        for i in t:
            i.set(name=prefix + i.__class__.__name__)

    return t


def preset_tonic(prefix="ton_"):
    mean = Mean()
    std = StDev()
    rng = Range()
    auc = AUC()

    t = [mean, std, rng, auc]

    if prefix is not None:
        for i in t:
            i.set(name=prefix + i.__class__.__name__)

    return t


def fmap(segments, algorithms, alt_signal=None):
    # TODO : rename extract_indicators
    """
    Generates a list composed of a list of results for each segment.

    [[result for each algorithm] for each segment]
    :param segments: An iterable of segments (e.g. an initialized SegmentGenerator)
    :param algorithms: A list of algorithms
    :param alt_signal: The signal that will be used instead of the one referenced in the segments

    :return: values, col_names A tuple: matrix (segment x algorithms) containing a value for each
     algorithm, the list of the algorithm names.
    """
    from numpy import asarray as _asarray
    values = _asarray([[seg.get_begin_time(), seg.get_end_time(), seg.get_label()] +
                       [alg(seg(alt_signal)) for alg in algorithms] for seg in (
        segments(alt_signal) if isinstance(segments, SegmentsGenerator) else segments
    )])
    col_names = ["begin", "end", "label"] + map(lambda x: x.__repr__(), algorithms)
    return values, _array(col_names)


def algo(function, **kwargs):
    """
    Builds on the fly a new algorithm class using the passed function and params if passed.
    Note that the cache will not be available.
    :param function: function(data, params) to be called
    :param kwargs: parameters to pass to the function.
    :return: An algorithm class if params is None else a parametrized algorithm instance.
    """

    from .BaseAlgorithm import Algorithm

    class Custom(Algorithm):
        def __init__(self, **kwargs):
            Algorithm.__init__(self, **kwargs)

        @classmethod
        def algorithm(cls, data, params):
            return function(data, params)

    if len(kwargs) == 0:
        return Custom
    else:
        return Custom(**kwargs)


def test():
    from pytest import main as m
    from os.path import dirname as d
    m(['-x', d(__file__)])
