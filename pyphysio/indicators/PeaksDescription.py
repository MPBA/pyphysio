# coding=utf-8
from __future__ import division

from abc import abstractmethod as _abstract, ABCMeta as _ABCMeta

import numpy as _np
from ..BaseIndicator import Indicator as _Indicator
from ..tools.Tools import PeakDetection as _PeakDetection, PeakSelection as _PeakSelection, Durations as _Durations, \
    Slopes as _Slopes

__author__ = 'AleB'


class _Peaks(_Indicator):
    """
    Peaks base class
    """
    __metaclass__ = _ABCMeta

    def __init__(self, delta, **kwargs):
        assert delta > 0, 'Parameter delta, i.e. amplitude of the minimum peak, has to be > 0'
        _Indicator.__init__(self, delta=delta, **kwargs)

    @classmethod
    @_abstract
    def algorithm(cls, data, params):
        pass


class PeaksMax(_Peaks):
    """
    Return the maximum amplitude of detected peaks.

    Parameters
    ----------
    delta : float, >0
        Minimum amplitude of peaks to be selected
    
    Returns
    -------
    mx : float
        Maximum amplitude of detected peaks
    
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
    Return the minimum amplitude of detected peaks.

    Parameters
    ----------
    delta : float, >0
        Minimum amplitude of peaks to be selected
    
    Returns
    -------
    mn : float
        Minimum amplitude of detected peaks
    
    """
    def __init__(self, delta, **kwargs):
        _Indicator.__init__(self, delta=delta, **kwargs)

    @classmethod
    def algorithm(cls, data, params):
        delta = params['delta']

        idx_maxs, idx_mins, val_maxs, val_mins = _PeakDetection(delta=delta)(data)

        if len(idx_maxs) == 0:
            cls.warn("No peak found, returning numpy.nan")
            return _np.nan
        else:
            return _np.nanmin(val_maxs)


class PeaksMean(_Peaks):
    """
    Return the average amplitude of detected peaks.

    Parameters
    ----------
    delta : float, >0
        Minimum amplitude of peaks to be selected
    
    Returns
    -------
    av : float
        Average amplitude of detected peaks
    
    """
    def __init__(self, delta, **kwargs):
        _Indicator.__init__(self, delta=delta, **kwargs)

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
    Return the number of detected peaks.

    Parameters
    ----------
    delta : float, >0
        Minimum amplitude of peaks to be selected
    
    Returns
    -------
    n : float
        Number of detected peaks
    
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
            return len(idx_maxs)


class _PeaksInterval(_Peaks):
    """
    Peaks base class
    """
    __metaclass__ = _ABCMeta

    def __init__(self, delta, win_pre=1, win_post=1, **kwargs):
        assert win_pre > 0, 'win_pre must be > 0'
        assert win_post > 0, 'win_post must be > 0'
        _Indicator.__init__(self, delta=delta, win_pre=win_pre, win_post=win_post, **kwargs)

    @classmethod
    @_abstract
    def algorithm(cls, data, params):
        pass


class DurationMin(_PeaksInterval):
    """
    Return the minimum duration of detected peaks.

    Parameters
    ----------
    delta : float, >0
        Minimum amplitude of peaks to be selected
    win_pre : float, >0, default=1
        Interval before a detected peak where to search the start of the peak
    win_post : float, >0, default=1
        Interval after a detected peak where to search the end of the peak
        
    Returns
    -------
    mn : float
        Minimum duration of detected peaks
    
    """
    def __init__(self, delta, win_pre=1, win_post=1, **kwargs):
        _Indicator.__init__(self, delta=delta, win_pre=win_pre, win_post=win_post, **kwargs)

    @classmethod
    def algorithm(cls, signal, params):
        delta = params['delta']
        win_pre = params['win_pre']
        win_post = params['win_post']

        idx_maxs, idx_mins, val_maxs, val_mins = _PeakDetection(delta=delta)(signal)
        if len(idx_maxs) == 0:
            cls.warn("No peaks found")
            return _np.nan

        idxs_start, idxs_stop = _PeakSelection(indices=idx_maxs, win_pre=win_pre, win_post=win_post)(signal)

        if len(idxs_start) == 0:
            cls.warn("Unable to detect the start of the peaks")
            return _np.nan
        else:
            durations = _Durations(starts=idxs_start, stops=idxs_stop)(signal)
            return _np.nanmin(_np.array(durations))


class DurationMax(_PeaksInterval):
    """
    Return the maximum duration of detected peaks.

    Parameters
    ----------
    delta : float, >0
        Minimum amplitude of peaks to be selected
    win_pre : float, >0, default=1
        Interval before a detected peak where to search the start of the peak
    win_post : float, >0, default=1
        Interval after a detected peak where to search the end of the peak
        
    Returns
    -------
    mx : float
        Maximum duration of detected peaks
    
    """
    def __init__(self, delta, win_pre=1, win_post=1, **kwargs):
        _Indicator.__init__(self, delta=delta, win_pre=win_pre, win_post=win_post, **kwargs)

    @classmethod
    def algorithm(cls, signal, params):

        delta = params['delta']
        win_pre = params['win_pre']
        win_post = params['win_post']

        idx_maxs, idx_mins, val_maxs, val_mins = _PeakDetection(delta=delta)(signal)
        if len(idx_maxs) == 0:
            cls.warn("No peaks found")
            return _np.nan

        idxs_start, idxs_stop = _PeakSelection(indices=idx_maxs, win_pre=win_pre, win_post=win_post)(signal)

        if len(idxs_start) == 0:
            cls.warn("Unable to detect the start of the peaks")
            return _np.nan
        else:
            durations = _Durations(starts=idxs_start, stops=idxs_stop)(signal)
            return _np.nanmax(_np.array(durations))


class DurationMean(_PeaksInterval):
    """
    Return the average duration of detected peaks.

    Parameters
    ----------
    delta : float, >0
        Minimum amplitude of peaks to be selected
    win_pre : float, >0, default=1
        Interval before a detected peak where to search the start of the peak
    win_post : float, >0, default=1
        Interval after a detected peak where to search the end of the peak
        
    Returns
    -------
    av : float
        Average duration of detected peaks
    
    """
    def __init__(self, delta, win_pre=1, win_post=1, **kwargs):
        _Indicator.__init__(self, delta=delta, win_pre=win_pre, win_post=win_post, **kwargs)

    @classmethod
    def algorithm(cls, signal, params):
        delta = params['delta']
        win_pre = params['win_pre']
        win_post = params['win_post']

        idx_maxs, idx_mins, val_maxs, val_mins = _PeakDetection(delta=delta)(signal)
        if len(idx_maxs) == 0:
            cls.warn("No peaks found")
            return _np.nan

        idxs_start, idxs_stop = _PeakSelection(indices=idx_maxs, win_pre=win_pre, win_post=win_post)(signal)

        if len(idxs_start) == 0:
            cls.warn("Unable to detect the start of the peaks")
            return _np.nan
        else:
            durations = _Durations(starts=idxs_start, stops=idxs_stop)(signal)
            return _np.nanmean(_np.array(durations))


class SlopeMin(_PeaksInterval):
    """
    Return the minimum slope of detected peaks.

    Parameters
    ----------
    delta : float, >0
        Minimum amplitude of peaks to be selected
    win_pre : float, >0, default=1
        Interval before a detected peak where to search the start of the peak
    win_post : float, >0, default=1
        Interval after a detected peak where to search the end of the peak
        
    Returns
    -------
    mn : float
        Minimum slope of detected peaks
    
    """
    def __init__(self, delta, win_pre=1, win_post=1, **kwargs):
        _Indicator.__init__(self, delta=delta, win_pre=win_pre, win_post=win_post, **kwargs)

    @classmethod
    def algorithm(cls, signal, params):
        delta = params['delta']
        win_pre = params['win_pre']
        win_post = params['win_post']

        idx_maxs, idx_mins, val_maxs, val_mins = _PeakDetection(delta=delta)(signal)
        if len(idx_maxs) == 0:
            cls.warn("No peaks found")
            return _np.nan

        idxs_start, idxs_stop = _PeakSelection(indices=idx_maxs, win_pre=win_pre, win_post=win_post)(signal)
        
        if len(idxs_start) == 0:
            cls.warn("Unable to detect the start of the peaks")
            return _np.nan
        else:
            slopes = _Slopes(starts=idxs_start, peaks=idx_maxs)(signal)
            return _np.nanmin(_np.array(slopes))


class SlopeMax(_PeaksInterval):
    """
    Return the maximum slope of detected peaks.

    Parameters
    ----------
    delta : float, >0
        Minimum amplitude of peaks to be selected
    win_pre : float, >0, default=1
        Interval before a detected peak where to search the start of the peak
    win_post : float, >0, default=1
        Interval after a detected peak where to search the end of the peak
        
    Returns
    -------
    mx : float
        Maximum slope of detected peaks
    
    """
    def __init__(self, delta, win_pre=1, win_post=1, **kwargs):
        _Indicator.__init__(self, delta=delta, win_pre=win_pre, win_post=win_post, **kwargs)

    @classmethod
    def algorithm(cls, signal, params):
        delta = params['delta']
        win_pre = params['win_pre']
        win_post = params['win_post']

        idx_maxs, idx_mins, val_maxs, val_mins = _PeakDetection(delta=delta)(signal)
        if len(idx_maxs) == 0:
            cls.warn("No peaks found")
            return _np.nan

        idxs_start, idxs_stop = _PeakSelection(indices=idx_maxs, win_pre=win_pre, win_post=win_post)(signal)
        
        if len(idxs_start) == 0:
            cls.warn("Unable to detect the start of the peaks")
            return _np.nan
        else:
            slopes = _Slopes(starts=idxs_start, peaks=idx_maxs)(signal)
            return _np.nanmax(_np.array(slopes))


class SlopeMean(_PeaksInterval):
    """
    Return the average slope of detected peaks.

    Parameters
    ----------
    delta : float, >0
        Minimum amplitude of peaks to be selected
    win_pre : float, >0, default=1
        Interval before a detected peak where to search the start of the peak
    win_post : float, >0, default=1
        Interval after a detected peak where to search the end of the peak
        
    Returns
    -------
    mx : float
        Maximum slope of detected peaks
    
    """
    def __init__(self, delta, win_pre=1, win_post=1, **kwargs):
        _Indicator.__init__(self, delta=delta, win_pre=win_pre, win_post=win_post, **kwargs)

    @classmethod
    def algorithm(cls, signal, params):
        delta = params['delta']
        win_pre = params['win_pre']
        win_post = params['win_post']

        idx_maxs, idx_mins, val_maxs, val_mins = _PeakDetection(delta=delta)(signal)
        if len(idx_maxs) == 0:
            cls.warn("No peaks found")
            return _np.nan

        idxs_start, idxs_stop = _PeakSelection(indices=idx_maxs, win_pre=win_pre, win_post=win_post)(signal)
        if len(idxs_start) == 0:
            cls.warn("Unable to detect the start of the peaks")
            return _np.nan
        else:
            slopes = _Slopes(starts=idxs_start, peaks=idx_maxs)(signal)
            return _np.nanmean(_np.array(slopes))
