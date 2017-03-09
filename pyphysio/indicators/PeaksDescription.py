# coding=utf-8
from __future__ import division

from abc import abstractmethod as _abstract, ABCMeta as _ABCMeta

import numpy as _np
from ..BaseIndicator import Indicator as _Indicator
from ..tools.Tools import PeakDetection as _PeakDetection, PeakSelection as _PeakSelection, Durations as _Durations, \
    Slopes as _Slopes
from ..Parameters import Parameter as _Par

__author__ = 'AleB'


class _Peaks(_Indicator):
    """
    Peaks base class
    """
    __metaclass__ = _ABCMeta

    def __init__(self, delta, **kwargs):
        _Indicator.__init__(self, delta=delta, **kwargs)

    @classmethod
    @_abstract
    def algorithm(cls, data, params):
        pass

    _params_descriptors = {
        'delta': _Par(2, float, 'Amplitude of the minimum peak (>0)', 0, lambda x: x > 0)
    }


class PeaksMax(_Peaks):
    """
    Peaks Max
    
    Description ...

    Parameters
    ----------
    
    Optional:
    degree : int, >0, default = 1
        Sample interval to compute the differences
    
    Returns
    -------
    signal : 
        Differences signal. 

    Notes
    -----
    Note that the length of the returned signal is the lenght of the input_signal minus degree.
    """

    def __init__(self, delta, **kwargs):
        _Indicator.__init__(self, delta=delta, **kwargs)

    @classmethod
    def algorithm(cls, signal, params):
        delta = params['delta']

        idx_maxs, idx_mins, val_maxs, val_mins = _PeakDetection(delta=delta)(signal)

        if len(idx_maxs) == 0:
            cls.warn("No peak found")
            return _np.nan
        else:
            return _np.nanmax(val_maxs)


class PeaksMin(_Peaks):
    """
    Peaks Min
    """

    @classmethod
    def algorithm(cls, data, params):
        delta = params['delta']

        idx_maxs, idx_mins, val_maxs, val_mins = _PeakDetection(delta=delta)(data)

        if len(idx_maxs) == 0:
            cls.warn("No peak found")
            return _np.nan
        else:
            return _np.nanmin(val_maxs)


class PeaksMean(_Peaks):
    """
    Peaks Mean
    """

    @classmethod
    def algorithm(cls, data, params):
        delta = params['delta']

        idx_maxs, idx_mins, val_maxs, val_mins = _PeakDetection(delta=delta)(data)

        if len(idx_maxs) == 0:
            cls.warn("No peak found")
            return _np.nan
        else:
            return _np.nanmean(val_maxs)


class PeaksNum(_Peaks):
    """
    Number of Peaks
    """

    @classmethod
    def algorithm(cls, signal, params):
        delta = params['delta']

        idx_maxs, idx_mins, val_maxs, val_mins = _PeakDetection(delta=delta)(signal)

        if len(idx_maxs) == 0:
            cls.warn("No peak found")
            return _np.nan
        else:
            return len(idx_maxs)


class _PeaksInterval(_Indicator):
    """
    Peaks base class
    """
    __metaclass__ = _ABCMeta

    def __init__(self, delta, pre_max, post_max, **kwargs):
        _Indicator.__init__(self, delta=delta, pre_max=pre_max, post_max=post_max, **kwargs)

    @classmethod
    @_abstract
    def algorithm(cls, data, params):
        pass

    _params_descriptors = {
        'delta': _Par(2, float, 'Amplitude of the minimum peak', 0, lambda x: x > 0),
        'pre_max': _Par(1, float,
                        'Duration (in seconds) of interval before the peak that is considered to find the start of the '
                        'peak (>0)',
                        1, lambda x: x > 0),
        'post_max': _Par(1, float,
                         'Duration (in seconds) of interval after the peak that is considered to find the start of the '
                         'peak (>0)',
                         1, lambda x: x > 0)
    }


class DurationMin(_PeaksInterval):
    """
    Min duration of Peaks
    """

    @classmethod
    def algorithm(cls, signal, params):
        delta = params['delta']
        pre_max = params['pre_max']
        post_max = params['post_max']

        idx_maxs, idx_mins, val_maxs, val_mins = _PeakDetection(delta=delta)(signal)
        if len(idx_maxs) == 0:
            cls.warn("No peaks found")
            return _np.nan

        idxs_start, idxs_stop = _PeakSelection(idx_max=idx_maxs, pre_max=pre_max, post_max=post_max)(signal)

        if len(idxs_start) == 0:
            cls.warn("Unable to detect the start of the peaks")
            return _np.nan
        else:
            durations = _Durations(starts=idxs_start, stops=idxs_stop)(signal)
            return _np.nanmin(_np.array(durations))


class DurationMax(_PeaksInterval):
    """
    Max duration of Peaks
    """

    @classmethod
    def algorithm(cls, signal, params):

        delta = params['delta']
        pre_max = params['pre_max']
        post_max = params['post_max']

        idx_maxs, idx_mins, val_maxs, val_mins = _PeakDetection(delta=delta)(signal)
        if len(idx_maxs) == 0:
            cls.warn("No peaks found")
            return _np.nan

        idxs_start, idxs_stop = _PeakSelection(idx_max=idx_maxs, pre_max=pre_max, post_max=post_max)(signal)

        if len(idxs_start) == 0:
            cls.warn("Unable to detect the start of the peaks")
            return _np.nan
        else:
            durations = _Durations(starts=idxs_start, stops=idxs_stop)(signal)
            return _np.nanmax(_np.array(durations))


class DurationMean(_PeaksInterval):
    """
    Mean duration of Peaks
    """

    @classmethod
    def algorithm(cls, signal, params):
        delta = params['delta']
        pre_max = params['pre_max']
        post_max = params['post_max']

        idx_maxs, idx_mins, val_maxs, val_mins = _PeakDetection(delta=delta)(signal)
        if len(idx_maxs) == 0:
            cls.warn("No peaks found")
            return _np.nan

        idxs_start, idxs_stop = _PeakSelection(idx_max=idx_maxs, pre_max=pre_max, post_max=post_max)(signal)

        if len(idxs_start) == 0:
            cls.warn("Unable to detect the start of the peaks")
            return _np.nan
        else:
            durations = _Durations(starts=idxs_start, stops=idxs_stop)(signal)
            return _np.nanmean(_np.array(durations))


class SlopeMin(_PeaksInterval):
    """
    Min slope of Peaks
    """

    @classmethod
    def algorithm(cls, signal, params):

        delta = params['delta']
        pre_max = params['pre_max']
        post_max = params['post_max']

        idx_maxs, idx_mins, val_maxs, val_mins = _PeakDetection(delta=delta)(signal)
        if len(idx_maxs) == 0:
            cls.warn("No peaks found")
            return _np.nan

        idxs_start, idxs_stop = _PeakSelection(idx_max=idx_maxs, pre_max=pre_max, post_max=post_max)(signal)
        if len(idxs_start) == 0:
            cls.warn("Unable to detect the start of the peaks")
            return _np.nan
        else:
            slopes = _Slopes(starts=idxs_start, peaks=idx_maxs)(signal)
            return _np.nanmin(_np.array(slopes))


class SlopeMax(_PeaksInterval):
    """
    Max slope of Peaks
    """

    @classmethod
    def algorithm(cls, signal, params):
        delta = params['delta']
        pre_max = params['pre_max']
        post_max = params['post_max']

        idx_maxs, idx_mins, val_maxs, val_mins = _PeakDetection(delta=delta)(signal)
        if len(idx_maxs) == 0:
            cls.warn("No peaks found")
            return _np.nan

        idxs_start, idxs_stop = _PeakSelection(idx_max=idx_maxs, pre_max=pre_max, post_max=post_max)(signal)
        if len(idxs_start) == 0:
            cls.warn("Unable to detect the start of the peaks")
            return _np.nan
        else:
            slopes = _Slopes(starts=idxs_start, peaks=idx_maxs)(signal)
            return _np.nanmax(_np.array(slopes))


class SlopeMean(_PeaksInterval):
    """
    Min slope of Peaks
    """

    @classmethod
    def algorithm(cls, signal, params):
        delta = params['delta']
        pre_max = params['pre_max']
        post_max = params['post_max']

        idx_maxs, idx_mins, val_maxs, val_mins = _PeakDetection(delta=delta)(signal)
        if len(idx_maxs) == 0:
            cls.warn("No peaks found")
            return _np.nan

        idxs_start, idxs_stop = _PeakSelection(idx_max=idx_maxs, pre_max=pre_max, post_max=post_max)(signal)
        if len(idxs_start) == 0:
            cls.warn("Unable to detect the start of the peaks")
            return _np.nan
        else:
            slopes = _Slopes(starts=idxs_start, peaks=idx_maxs)(signal)
            return _np.nanmean(_np.array(slopes))
