# coding=utf-8
from __future__ import division

import numpy as _np
from ..BaseIndicator import Indicator as _Indicator
from ..Utility import PhUI as _PhUI
from ..tools.Tools import PeakDetection as _PeakDetection, PeakSelection as _PeakSelection
from ..Parameters import Parameter as _Par

__author__ = 'AleB'


class PeaksMax(_Indicator):
    @classmethod
    def algorithm(cls, data, params):
        """
        Peaks Max
        """

        maxs, mins = _PeakDetection(params['delta'])(data)

        if _np.shape(maxs)[0] == 0:
            _PhUI.w("_np.shape(maxs)[0] == 0")  # TODO: Put a more explicative message
            return _np.nan
        else:
            return _np.nanmax(maxs[:, 1])

    _params_descriptors = {
        'delta': _Par(2, float, 'Amplitude of the minimum peak', 0, lambda x: x > 0)
    }


class PeaksMin(_Indicator):
    @classmethod
    def algorithm(cls, data, params):
        """
        Peaks Min
        """

        maxs, mins = _PeakDetection(params['delta'])(data)
        if _np.shape(maxs)[0] == 0:
            _PhUI.w("_np.shape(maxs)[0] == 0")  # TODO: Put a more explicative message
            return _np.nan
        else:
            return _np.nanmin(maxs[:, 1])

    _params_descriptors = {
        'delta': _Par(2, float, 'Amplitude of the minimum peak', 0, lambda x: x > 0)
    }


class PeaksMean(_Indicator):
    @classmethod
    def algorithm(cls, data, params):
        """
        Peaks Mean
        """

        maxs, mins = _PeakDetection(params['delta'])(data)

        if _np.shape(maxs)[0] == 0:
            _PhUI.w("_np.shape(maxs)[0] == 0")  # TODO: Put a more explicative message
            return _np.nan
        else:
            return _np.nanmean(maxs[:, 1])

    _params_descriptors = {
        'delta': _Par(2, float, 'Amplitude of the minimum peak', 0, lambda x: x > 0)
    }


class PeaksNum(_Indicator):
    @classmethod
    def algorithm(cls, data, params):
        """
        Number of Peaks
        """

        maxs, mins = _PeakDetection(params['delta'])(data)

        if _np.shape(maxs)[0] == 0:
            _PhUI.w("_np.shape(maxs)[0] == 0")  # TODO: Put a more explicative message
            return _np.nan
        else:
            return len(maxs[:, 1])

    _params_descriptors = {
        'delta': _Par(2, float, 'Amplitude of the minimum peak', 0, lambda x: x > 0)
    }


class DurationMin(_Indicator):
    """
    Min duration of Peaks
    """

    @classmethod
    def algorithm(cls, data, params):
        maxs, mins = _PeakDetection(params['delta'])(data)
        idxs_start, idxs_stop = _PeakSelection(maxs=maxs, pre_max=params['pre_max'],
                                               post_max=params['post_max'])(data)

        if len(idxs_start) == 0:
            _PhUI.w("len(idxs_start) == 0")
            return _np.nan
        else:
            fsamp = data.get_sampling_freq()
            durations = []
            for I in range(len(idxs_start)):
                if (_np.isnan(idxs_stop[I]) is False) & (_np.isnan(idxs_start[I]) is False):
                    durations.append((idxs_stop[I] - idxs_start[
                        I]) / fsamp)  # TODO: volendo si puo mettere in cache anche il calcolo di durations
                else:
                    durations.append(_np.nan)
            return _np.nanmin(_np.array(durations))

    _params_descriptors = {
        'delta': _Par(2, float, 'Amplitude of the minimum peak', 0, lambda x: x > 0),
        'pre_max': _Par(2, float,
                        'Duration (in seconds) of interval before the peak that is considered to find the start of the peak',
                        1, lambda x: x > 0),
        'post_max': _Par(2, float,
                         'Duration (in seconds) of interval after the peak that is considered to find the start of the peak',
                         1, lambda x: x > 0)
    }


class DurationMax(_Indicator):
    """
    Max duration of Peaks
    """

    @classmethod
    def algorithm(cls, data, params):
        maxs, mins = _PeakDetection(delta=params['delta'])(data)
        idxs_start, idxs_stop = _PeakSelection(maxs=maxs, pre_max=params['pre_max'],
                                               post_max=params['post_max'])(data)

        if len(idxs_start) == 0:
            _PhUI.w("len(idxs_start) == 0")
            return _np.nan
        else:
            fsamp = data.get_sampling_freq()
            durations = []
            for I in range(len(idxs_start)):
                if (_np.isnan(idxs_stop[I]) is False) & (_np.isnan(idxs_start[I]) is False):
                    durations.append((idxs_stop[I] - idxs_start[
                        I]) / fsamp)  # TODO: volendo si puo mettere in cache anche il calcolo di durations
                else:
                    durations.append(_np.nan)

            return _np.nanmax(_np.array(durations))

    _params_descriptors = {
        'delta': _Par(2, float, 'Amplitude of the minimum peak', 0, lambda x: x > 0),
        'pre_max': _Par(2, float,
                        'Duration (in seconds) of interval before the peak that is considered to find the start of the peak',
                        1, lambda x: x > 0),
        'post_max': _Par(2, float,
                         'Duration (in seconds) of interval after the peak that is considered to find the start of the peak',
                         1, lambda x: x > 0)
    }


class DurationMean(_Indicator):
    """
    Mean duration of Peaks
    """

    @classmethod
    def algorithm(cls, data, params):
        maxs, mins = _PeakDetection(delta=params['delta'])(data)
        idxs_start, idxs_stop = _PeakSelection(maxs=maxs, pre_max=params['pre_max'],
                                               post_max=params['post_max'])(data)

        if len(idxs_start) == 0:
            _PhUI.w("len(idxs_start) == 0")
            return _np.nan
        else:
            fsamp = data.get_sampling_freq()
            durations = []
            for I in range(len(idxs_start)):
                if (_np.isnan(idxs_stop[I]) is False) & (_np.isnan(idxs_start[I]) is False):
                    durations.append((idxs_stop[I] - idxs_start[
                        I]) / fsamp)  # TODO: volendo si puo mettere in cache anche il calcolo di durations
                else:
                    durations.append(_np.nan)
            return _np.nanmean(_np.array(durations))

    _params_descriptors = {
        'delta': _Par(2, float, 'Amplitude of the minimum peak', 0, lambda x: x > 0),
        'pre_max': _Par(2, float,
                        'Duration (in seconds) of interval before the peak that is considered to find the start of the peak',
                        1, lambda x: x > 0),
        'post_max': _Par(2, float,
                         'Duration (in seconds) of interval after the peak that is considered to find the start of the peak',
                         1, lambda x: x > 0)
    }


class SlopeMin(_Indicator):
    """
    Min slope of Peaks
    """

    @classmethod
    def algorithm(cls, data, params):
        maxs, mins = _PeakDetection(delta=params['delta'])(data)
        idxs_start, idxs_stop = _PeakSelection(maxs=maxs, pre_max=params['pre_max'],
                                               post_max=params['post_max'])(data)

        if len(idxs_start) == 0:
            _PhUI.w("len(idxs_start) == 0")
            return _np.nan
        else:
            fsamp = data.get_sampling_freq()
            idxs_peak = maxs[:, 0]
            slopes = []
            for I in range(len(idxs_start)):
                if (_np.isnan(idxs_peak[I]) is False) & (_np.isnan(idxs_start[I]) is False):
                    dy = data[idxs_peak[I]] - data[idxs_start[I]]
                    dt = (idxs_peak[I] - idxs_start[I]) / fsamp
                    slopes.append(dy / dt)  # TODO: volendo si puo mettere in cache anche il calcolo delle slopes
                else:
                    slopes.append(_np.nan)  # TODO: volendo si puo mettere in cache anche il calcolo delle slopes
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
    def algorithm(cls, data, params):
        maxs, mins = _PeakDetection(delta=params['delta'])(data)
        idxs_start, idxs_stop = _PeakSelection(maxs=maxs, pre_max=params['pre_max'],
                                               post_max=params['post_max'])(data)

        if len(idxs_start) == 0:
            _PhUI.w("len(idxs_start) == 0")
            return _np.nan
        else:
            fsamp = data.get_sampling_freq()
            idxs_peak = maxs[:, 0]
            slopes = []
            for I in range(len(idxs_start)):
                if (_np.isnan(idxs_peak[I]) == False) & (_np.isnan(idxs_start[I]) == False):
                    dy = data[idxs_peak[I]] - data[idxs_start[I]]
                    dt = (idxs_peak[I] - idxs_start[I]) / fsamp
                    slopes.append(dy / dt)  # TODO: volendo si puo mettere in cache anche il calcolo delle slopes
                else:
                    slopes.append(_np.nan)  # TODO: volendo si puo mettere in cache anche il calcolo delle slopes
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
    def algorithm(cls, data, params):
        maxs, mins = _PeakDetection(delta=params['delta'])(data)
        idxs_start, idxs_stop = _PeakSelection(maxs=maxs, pre_max=params['pre_max'],
                                               post_max=params['post_max'])(data)

        if len(idxs_start) == 0:
            _PhUI.w("len(idxs_start) == 0")
            return _np.nan
        else:
            fsamp = data.get_sampling_freq()
            idxs_peak = maxs[:, 0]
            slopes = []
            for I in range(len(idxs_start)):
                if (_np.isnan(idxs_peak[I]) is False) & (_np.isnan(idxs_start[I]) is False):
                    dy = data[idxs_peak[I]] - data[idxs_start[I]]
                    dt = (idxs_peak[I] - idxs_start[I]) / fsamp
                    slopes.append(dy / dt)  # TODO: volendo si puo mettere in cache anche il calcolo delle slopes
                else:
                    slopes.append(_np.nan)  # TODO: volendo si puo mettere in cache anche il calcolo delle slopes
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
