# coding=utf-8
from __future__ import division
import pyphysio.filters.Filters
from pyphysio.indicators.TimeDomain import *
from pyphysio.indicators.FrequencyDomain import *
from pyphysio.indicators.NonLinearDomain import *
from pyphysio.indicators.PeaksDescription import *
from pyphysio.filters.Filters import *
from pyphysio.estimators.Estimators import *
from pyphysio.tools.Tools import *
from pyphysio.segmentation.SegmentsGenerators import *
import pyphysio.segmentation.SegmentsGenerators
from pyphysio.BaseSegmentation import Segment
from pyphysio.Signal import EvenlySignal, UnevenlySignal
from numpy import nan as _nan

__author__ = "AleB"


def compute_hrv(ibi):
    labels = ['VLF', 'LF', 'HF', 'rmssd', 'sdsd', 'RRmean', 'RRstd', 'RRmedian', 'pnn10', 'pnn25',
              'pnn50', 'mn', 'mx', 'sd1', 'sd2', 'sd12', 'sdell', 'DFA1', 'DFA2']

    hrv = []

    if len(ibi) >= 10:  # require at least 10 beats
        # initialize Frequency domain indicators
        VLF = PowerInBand(interp_freq=4, freq_max=0.04, freq_min=0.00001, method='ar')
        LF = PowerInBand(interp_freq=4, freq_max=0.15, freq_min=0.04, method='ar')
        HF = PowerInBand(interp_freq=4, freq_max=0.4, freq_min=0.15, method='ar')
        Total = PowerInBand(interp_freq=4, freq_max=2, freq_min=0.00001, method='ar')

        FD_indexes = [VLF, LF, HF, Total]

        # compute Frequency domain indicators and normalize
        tmp = []
        for x in FD_indexes:
            curr_idx = x(ibi)
            tmp.append(curr_idx)
        for i in range(3):
            hrv.append(tmp[i] / tmp[3])
    else:
        for i in range(4):
            hrv.append(_nan)

    # initialize Time domain + non-linear indicators
    rmssd = RMSSD()
    sdsd = SDSD()
    RRmean = Mean()
    RRstd = StDev()
    RRmedian = Median()

    pnn10 = PNNx(threshold=10)
    pnn25 = PNNx(threshold=25)
    pnn50 = PNNx(threshold=50)

    mn = Min()
    mx = Max()

    sd1 = PoincareSD1()
    sd2 = PoincareSD2()
    sd12 = PoincareSD1SD2()
    sdell = PoinEll()

    DFA1 = DFAShortTerm()
    DFA2 = DFALongTerm()

    #        triang = Triang()
    #        TINN = TINN()

    TD_indexes = [rmssd, sdsd, RRmean, RRstd, RRmedian, pnn10, pnn25, pnn50, mn, mx, sd1, sd2, sd12, sdell, DFA1, DFA2]
    # compute Time domain + non-linear indicators
    for x in TD_indexes:
        curr_idx = x(ibi)
        hrv.append(curr_idx)
    return hrv, labels


def compute_pha_ton_indicators(phasic, driver, delta):
    mean = Mean()
    std = StDev()
    rng = Range()

    pks_max = PeaksMax(delta=delta)
    pks_min = PeaksMin(delta=delta)
    pks_mean = PeaksMean(delta=delta)
    n_peaks = PeaksNum(delta=delta)

    dur_mean = DurationMean(delta=0.1, pre_max=2, post_max=2)

    slopes_mean = SlopeMean(delta=0.1, pre_max=2, post_max=2)

    auc = AUC()

    phasic_indexes = [mean, std, rng, pks_max, pks_min, pks_mean, n_peaks, dur_mean, slopes_mean, auc]
    tonic_indexes = [mean, std, rng, auc]

    labels = ['pha_mean', 'pha_std', 'pha_rng', 'pks_max', 'pks_min', 'pks_mean', 'n_peaks', 'dur_mean', 'slopes_mean',
              'pha_auc', 'ton_mean', 'ton_std', 'ton_rng', 'ton_auc']

    indicators = []

    for x in phasic_indexes:
        curr_ind = x(phasic)
        indicators.append(curr_ind)

    tonic = driver - phasic

    for x in tonic_indexes:
        curr_ind = x(tonic)
        indicators.append(curr_ind)

    return indicators, labels


def fmap(segments, algorithms, alt_signal=None):
    # TODO : rename extract_indicators
    """
    Generates a list composed of a list of results for each segment.

    [[result for each algorithm] for each segment]
    :param segments: An iterable of segments (e.g. an initialized SegmentGenerator)
    :param algorithms: A list of algorithms
    :param alt_signal: The signal that will be used instead of the one referenced in the segments
    
    :return: values, labels, col_names A tuple: a list containing a list for each segment containing a value for each algorithm, the list of the
    algorithm names.
    """
    from numpy import asarray as _asarray
    values = _asarray([[seg.get_begin_time(), seg.get_end_time()] + [alg(seg(alt_signal)) for alg in algorithms]
                       for seg in segments])
    labels = _asarray([seg.get_label() for seg in segments])
    col_names = ["begin", "end"] + map(lambda x: x.__repr__(), algorithms)
    return values, labels, col_names


def algo(function, **kwargs):
    """
    Builds on the fly a new algorithm class using the passed function and params if passed.
    Note that the cache will not be available.
    :param function: function(data, params) to be called
    :param kwargs: parameters to pass to the function.
    :return: An algorithm class if params is None else a parametrized algorithm instance.
    """

    from BaseAlgorithm import Algorithm

    class Custom(Algorithm):
        @classmethod
        def algorithm(cls, data, params):
            return function(data, params)

    if len(kwargs) == 0:
        return Custom
    else:
        return Custom(**kwargs)
