# coding=utf-8
from __future__ import division

import numpy as _np
from ..BaseIndicator import Indicator as _Indicator
from ..Utility import PhUI as _PhUI
from ..tools.Tools import PeakDetection as _PeakDetection, PeakSelection as _PeakSelection, Durations as _Durations, Slopes as _Slopes
from ..Parameters import Parameter as _Par

__author__ = 'AleB'


class PeaksMax(_Indicator):
    @classmethod
    def algorithm(cls, signal, params):
        """
        Peaks Max
        """
        delta = params['delta']

        idx_maxs, idx_mins, val_maxs, val_mins = _PeakDetection(delta=delta)(signal)

        if len(idx_maxs) == 0:
            cls.warn("No peak found")
            return _np.nan
        else:
            return _np.nanmax(val_maxs)

    _params_descriptors = {
        'delta': _Par(2, float, 'Amplitude of the minimum peak', 0, lambda x: x > 0)
    }


class PeaksMin(_Indicator):
    @classmethod
    def algorithm(cls, data, params):
        """
        Peaks Min
        """
        delta = params['delta']

        idx_maxs, idx_mins, val_maxs, val_mins = _PeakDetection(delta=delta)(data)

        if len(idx_maxs) == 0:
            cls.warn("No peak found")
            return _np.nan
        else:
            return _np.nanmin(val_maxs)

    _params_descriptors = {
        'delta': _Par(2, float, 'Amplitude of the minimum peak', 0, lambda x: x > 0)
    }


class PeaksMean(_Indicator):
    @classmethod
    def algorithm(cls, data, params):
        """
        Peaks Mean
        """
        delta = params['delta']

        idx_maxs, idx_mins, val_maxs, val_mins = _PeakDetection(delta=delta)(data)

        if len(idx_maxs) == 0:
            cls.warn("No peak found")
            return _np.nan
        else:
            return _np.nanmean(val_maxs)

    _params_descriptors = {
        'delta': _Par(2, float, 'Amplitude of the minimum peak', 0, lambda x: x > 0)
    }


class PeaksNum(_Indicator):
    @classmethod
    def algorithm(cls, signal, params):
        """
        Number of Peaks
        """
        delta = params['delta']

        idx_maxs, idx_mins, val_maxs, val_mins = _PeakDetection(delta=delta)(signal)

        if len(idx_maxs) == 0:
            cls.warn("No peak found")
            return _np.nan
        else:
            return len(idx_maxs)

    _params_descriptors = {
        'delta': _Par(2, float, 'Amplitude of the minimum peak', 0, lambda x: x > 0)
    }


class DurationMin(_Indicator):
    """
    Min duration of Peaks
    """

    @classmethod
    def algorithm(cls, signal, params):
        delta = params['delta']
        pre_max = params['pre_max']
        post_max = params['post_max']

        idx_maxs, idx_mins, val_maxs, val_mins = _PeakDetection(delta=delta)(signal)
        if len(idx_maxs)==0:
            cls.warn("No peaks found")
            return _np.nan
            
        idxs_start, idxs_stop = _PeakSelection(maxs=idx_maxs, pre_max=pre_max, post_max=post_max)(signal)

        if len(idxs_start) == 0:
            cls.warn("Unable to detect the start of the peaks")
            return _np.nan
        else:
            durations = _Durations(starts=idxs_start, stops=idxs_stop)(signal)
            return _np.nanmin(_np.array(durations))

    _params_descriptors = {
        'delta': _Par(2, float, 'Amplitude of the minimum peak', 0, lambda x: x > 0),
        'pre_max': _Par(1, float,
                        'Duration (in seconds) of interval before the peak that is considered to find the start of the peak',
                        1, lambda x: x > 0),
        'post_max': _Par(1, float,
                         'Duration (in seconds) of interval after the peak that is considered to find the start of the peak',
                         1, lambda x: x > 0)
    }


class DurationMax(_Indicator):
    """
    Max duration of Peaks
    """

    @classmethod
    def algorithm(cls, signal, params):
        
        delta = params['delta']
        pre_max = params['pre_max']
        post_max = params['post_max']

        idx_maxs, idx_mins, val_maxs, val_mins = _PeakDetection(delta=delta)(signal)
        if len(idx_maxs)==0:
            cls.warn("No peaks found")
            return _np.nan
            
        idxs_start, idxs_stop = _PeakSelection(maxs=idx_maxs, pre_max=pre_max, post_max=post_max)(signal)

        if len(idxs_start) == 0:
            cls.warn("Unable to detect the start of the peaks")
            return _np.nan
        else:
            durations = _Durations(starts=idxs_start, stops=idxs_stop)(signal)
            return _np.nanmax(_np.array(durations))

    _params_descriptors = {
        'delta': _Par(2, float, 'Amplitude of the minimum peak', 0, lambda x: x > 0),
        'pre_max': _Par(1, float,
                        'Duration (in seconds) of interval before the peak that is considered to find the start of the peak',
                        1, lambda x: x > 0),
        'post_max': _Par(1, float,
                         'Duration (in seconds) of interval after the peak that is considered to find the start of the peak',
                         1, lambda x: x > 0)
    }


class DurationMean(_Indicator):
    """
    Mean duration of Peaks
    """

    @classmethod
    def algorithm(cls, signal, params):
        delta = params['delta']
        pre_max = params['pre_max']
        post_max = params['post_max']

        idx_maxs, idx_mins, val_maxs, val_mins = _PeakDetection(delta=delta)(signal)
        if len(idx_maxs)==0:
            cls.warn("No peaks found")
            return _np.nan
            
        idxs_start, idxs_stop = _PeakSelection(maxs=idx_maxs, pre_max=pre_max, post_max=post_max)(signal)

        if len(idxs_start) == 0:
            cls.warn("Unable to detect the start of the peaks")
            return _np.nan
        else:
            durations = _Durations(starts=idxs_start, stops=idxs_stop)(signal)
            return _np.nanmean(_np.array(durations))

    _params_descriptors = {
        'delta': _Par(2, float, 'Amplitude of the minimum peak', 0, lambda x: x > 0),
        'pre_max': _Par(1, float,
                        'Duration (in seconds) of interval before the peak that is considered to find the start of the peak',
                        1, lambda x: x > 0),
        'post_max': _Par(1, float,
                         'Duration (in seconds) of interval after the peak that is considered to find the start of the peak',
                         1, lambda x: x > 0)
    }


class SlopeMin(_Indicator):
    """
    Min slope of Peaks
    """

    @classmethod
    def algorithm(cls, signal, params):
        
        delta = params['delta']
        pre_max = params['pre_max']
        post_max = params['post_max']

        idx_maxs, idx_mins, val_maxs, val_mins = _PeakDetection(delta=delta)(signal)
        if len(idx_maxs)==0:
            cls.warn("No peaks found")
            return _np.nan
            
        idxs_start, idxs_stop = _PeakSelection(maxs=idx_maxs, pre_max=pre_max, post_max=post_max)(signal)     
        if len(idxs_start) == 0:
            cls.warn("Unable to detect the start of the peaks")
            return _np.nan
        else:
            slopes = _Slopes(starts=idxs_start, peaks=idx_maxs)(signal)
            return _np.nanmin(_np.array(slopes))

    _params_descriptors = {
        'delta': _Par(2, float, 'Amplitude of the minimum peak', 0, lambda x: x > 0),
        'pre_max': _Par(2, float,
                        'Duration (in seconds) of interval before the peak that is considered to find the start of the peak',
                        1, lambda x: x > 0),
        'post_max': _Par(2, float,
                         'Duration (in seconds) of interval after the peak that is considered to find the start of the peak',
                         1, lambda x: x > 0)
    }


class SlopeMax(_Indicator):
    """
    Max slope of Peaks
    """

    @classmethod
    def algorithm(cls, signal, params):
        delta = params['delta']
        pre_max = params['pre_max']
        post_max = params['post_max']

        idx_maxs, idx_mins, val_maxs, val_mins = _PeakDetection(delta=delta)(signal)
        if len(idx_maxs)==0:
            cls.warn("No peaks found")
            return _np.nan
            
        idxs_start, idxs_stop = _PeakSelection(maxs=idx_maxs, pre_max=pre_max, post_max=post_max)(signal)     
        if len(idxs_start) == 0:
            cls.warn("Unable to detect the start of the peaks")
            return _np.nan
        else:
            slopes = _Slopes(starts=idxs_start, peaks=idx_maxs)(signal)
            return _np.nanmax(_np.array(slopes))

    _params_descriptors = {
        'delta': _Par(2, float, 'Amplitude of the minimum peak', 0, lambda x: x > 0),
        'pre_max': _Par(2, float,
                        'Duration (in seconds) of interval before the peak that is considered to find the start of the peak',
                        1, lambda x: x > 0),
        'post_max': _Par(2, float,
                         'Duration (in seconds) of interval after the peak that is considered to find the start of the peak',
                         1, lambda x: x > 0)
    }


class SlopeMean(_Indicator):
    """
    Min slope of Peaks
    """

    @classmethod
    def algorithm(cls, signal, params):
        delta = params['delta']
        pre_max = params['pre_max']
        post_max = params['post_max']

        idx_maxs, idx_mins, val_maxs, val_mins = _PeakDetection(delta=delta)(signal)
        if len(idx_maxs)==0:
            cls.warn("No peaks found")
            return _np.nan
            
        idxs_start, idxs_stop = _PeakSelection(maxs=idx_maxs, pre_max=pre_max, post_max=post_max)(signal)     
        if len(idxs_start) == 0:
            cls.warn("Unable to detect the start of the peaks")
            return _np.nan
        else:
            slopes = _Slopes(starts=idxs_start, peaks=idx_maxs)(signal)
            return _np.nanmean(_np.array(slopes))

    _params_descriptors = {
        'delta': _Par(2, float, 'Amplitude of the minimum peak', 0, lambda x: x > 0),
        'pre_max': _Par(2, float,
                        'Duration (in seconds) of interval before the peak that is considered to find the start of the peak',
                        1, lambda x: x > 0),
        'post_max': _Par(2, float,
                         'Duration (in seconds) of interval after the peak that is considered to find the start of the peak',
                         1, lambda x: x > 0)
    }
