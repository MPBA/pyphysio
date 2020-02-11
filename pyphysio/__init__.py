# coding=utf-8
from __future__ import division

from numpy import array as _array
from .tools.Tools import *
import numpy as _np
from .filters import Filters
from .segmentation import SegmentsGenerators
from .indicators import FrequencyDomain
from .indicators import NonLinearDomain
from .indicators import PeaksDescription
from .indicators import TimeDomain
from .BaseSegmentation import Segment
from .Signal import EvenlySignal, UnevenlySignal, MultiEvenly, from_pickle, from_pickleable
from .interactive import Annotate
# BE CAREFUL with NAMES!!!
from .estimators.Estimators import *
from .filters.Filters import *
from .indicators.FrequencyDomain import *
from .indicators.NonLinearDomain import *
from .indicators.PeaksDescription import *
from .indicators.TimeDomain import *
from .sqi.SignalQuality import *
#from .tests import TestData
from .segmentation.SegmentsGenerators import *

#TODO: all signals as N_SAMPLES x N_CH, with N_CH =1 for non MultiEvenly

print("Please cite:")
print("Bizzego et al. (2019) 'pyphysio: A physiological signal processing library for data science approaches in physiology', SoftwareX")

__author__ = "AleB"

def nature2type(data):
    data.ph['signal_type'] = data.ph['signal_nature']
    if isinstance(data, NIRS):
        stim = data.get_stim()
        stim.ph['signal_type'] = stim.ph['signal_nature']
        data.set_stim(stim)
    return(data)
    
def preset_sqi_ecg(prefix="SQI_", method='ar'):
    K = Kurtosis(name='kurtosis')
    SPR = SpectralPowerRatio(method, name='SPR')
    DE = DerivativeEnergy(name='DE')
    
    t = [K, SPR, DE]

    if prefix is not None:
        for i in t:
            i.set(name=prefix + i.get("name"))

    return t

def preset_hrv_fd(prefix="IBI_", method='ar'):
    VLF = PowerInBand(interp_freq=4, freq_max=0.04, freq_min=0.00001, method=method, name="VLF_Pow")
    LF = PowerInBand(interp_freq=4, freq_max=0.15, freq_min=0.04, method=method, name="LF_Pow")
    HF = PowerInBand(interp_freq=4, freq_max=0.4, freq_min=0.15, method=method, name="HF_Pow")
    Total = PowerInBand(interp_freq=4, freq_max=2, freq_min=0.00001, method=method, name="Total_Pow")
    
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

def preset_eeg(prefix="eeg_", method='welch'):
    delta = PowerInBand(freq_min=0, freq_max=3, method=method, name="delta")
    theta = PowerInBand(freq_min=3.5, freq_max=7.5, method=method, name="theta")
    alpha = PowerInBand(freq_min=7.5, freq_max=13, method=method, name="alpha")
    beta = PowerInBand(freq_min=14, freq_max=30, method=method, name="beta")
    total = PowerInBand(freq_min=0, freq_max=100, method=method, name="total")
    
    t = [delta, theta, alpha, beta, total]

    if prefix is not None:
        for i in t:
            i.set(name=prefix + i.get("name"))

    return t


def preset_emg(prefix='emg_', method = 'welch'):
    mx = Max(name='maximum')
    mn = Min(name='minimum')
    mean = Mean(name='mean')
    rng = Range(name='range')
    sd = StDev(name='sd')
    auc = AUC(name='auc')
    en4_40 = PowerInBand(freq_min=4, freq_max=40, method=method, name="en_4_40")
    
    t = [mx, mn, mean, rng, sd, auc, en4_40]

    if prefix is not None:
        for i in t:
            i.set(name=prefix + i.get("name"))

    return t


def preset_resp(prefix='resp', method='welch'):
    e_low = PowerInBand(freq_min=0, freq_max=0.25, method=method, name="energy_low")
    e_high = PowerInBand(freq_min=0.25, freq_max=5, method=method, name="energy_high")
    resp_rate = PeakInBand(freq_min=0.25, freq_max=5, method=method, name="resp_rate")
    
    t = [e_low, e_high, resp_rate]
    
    if prefix is not None:
        for i in t:
            i.set(name=prefix + i.get("name"))

    return t

def preset_activity(prefix='activity', method='welch'):
    mx = Max(name='maximum')
    mn = Min(name='minimum')
    mean = Mean(name='mean')
    rng = Range(name='range')
    sd = StDev(name='sd')
    auc = AUC(name='auc')
    en_25 = PowerInBand(freq_min=0, freq_max=25, method=method, name="en_25")
    
    t = [mx, mn, mean, rng, sd, auc, en_25]

    if prefix is not None:
        for i in t:
            i.set(name=prefix + i.get("name"))

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
    
    seg_for = segments(alt_signal) if isinstance(segments, SegmentsGenerator) else segments
    
    values = []
    for seg in seg_for:
        segment_data = _np.array([seg.get_begin_time(), seg.get_end_time(), seg.get_label()]).reshape(3,1)
        vals_segment = []
        for alg in algorithms:
            vals_alg = _np.array(alg(seg(alt_signal)))

            if not alt_signal.is_multi():
#                vals_alg = _np.expand_dims([vals_alg], 1)
                vals_alg = _np.array([vals_alg])
            vals_segment.append(vals_alg)
            
        vals_segment = _np.array(vals_segment)
        seg_data_array = _np.repeat(segment_data, alt_signal.get_nchannels(), axis = 1)
        vals_segment = _np.concatenate([seg_data_array, vals_segment], axis = 0)
        values.append(vals_segment)
    
    values = _np.array(values)
    
    #for compatibility
    if not alt_signal.is_multi():
        values = values[:,:,0]
        
    col_names = ["begin", "end", "label"] + [x.__repr__() for x in algorithms]
    
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
