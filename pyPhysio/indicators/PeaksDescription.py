# coding=utf-8
from __future__ import division

from ..BaseIndicator import Indicator as _Indicator
from ..filters.Filters import Diff as _Diff
from ..tools.Tools import PeakDetection as _PeakDetection, PeakSelection as _PeakSelection
import numpy as _np

__author__ = 'AleB'

class PeaksMax(_Indicator):
    @classmethod
    def algorithm(cls, data, params):
        """
        Peaks Max
        """
        
        maxs, mins = _PeakDetection.get(data, delta=params['delta'])
        if np.shape(maxs)[0] == 0:
			#WARNING
			return _np.nan
		else:
			return _np.nanmax(maxs[:,1])

	@classmethod
    def check_params(cls, params):
        params = {
			'delta': FloatPar(0, 2, 'Amplitude of the minimum peak', '>0')
			}
        return params


class PeaksMin(_Indicator):
    @classmethod
    def algorithm(cls, data, params):
        """
        Peaks Min
        """
        
        maxs, mins = _PeakDetection.get(data, delta=params['delta'])
        if np.shape(maxs)[0] == 0:
			#WARNING
			return _np.nan
		else:
			return _np.nanmin(maxs[:,1])

	@classmethod
    def check_params(cls, params):
        params = {
			'delta': FloatPar(0, 2, 'Amplitude of the minimum peak', '>0')
			}
        return params


class PeaksMean(_Indicator):
    @classmethod
    def algorithm(cls, data, params):
        """
        Peaks Mean
        """
        
        maxs, mins = _PeakDetection.get(data, delta=params['delta'])
        if np.shape(maxs)[0] == 0:
			#WARNING
			return _np.nan
		else:
			return _np.nanmean(maxs[:,1])

	@classmethod
    def check_params(cls, params):
        params = {
			'delta': FloatPar(0, 2, 'Amplitude of the minimum peak', '>0')
			}
        return params


class PeaksNum(_Indicator):
    @classmethod
    def algorithm(cls, data, params):
        """
        Number of Peaks
        """
        
        maxs, mins = _PeakDetection.get(data, delta=params['delta'])
        if np.shape(maxs)[0] == 0:
			#WARNING
			return _np.nan
		else:
			return len(maxs[:,1])

	@classmethod
    def check_params(cls, params):
        params = {
			'delta': FloatPar(0, 2, 'Amplitude of the minimum peak', '>0')
			}
        return params


class DurationMin(_Indicator):
	"""
    Min duration of Peaks
    """
    @classmethod
    def algorithm(cls, data, params):
		maxs, mins = _PeakDetection.get(data, delta=params['delta'])
        idxs_start, idxs_stop = _PeakSelection.get(data, maxs=maxs, pre_max = params['pre_max'], post_max = params['post_max'])
        
        if len(idxs_start) == 0:
			#WARNING
			return _np.nan
		else:
			fsamp = data.sampling_freq
			durations = []
			for I in range(len(idxs_start)):
				if (_np.isnan(idxs_stop[I]) == False) & (_np.isnan(idxs_start[I]) == False):
					durations.append( (idxs_stop[I] - idxs_start[I]) / fsamp ) # TODO: volendo si puo mettere in cache anche il calcolo di durations
				else:
					durations.append(_np.nan)
			return(_np.nanmin(_np.array(durations)))

	@classmethod
    def check_params(cls, params):
        params = {
			'delta': FloatPar(0, 2, 'Amplitude of the minimum peak', '>0'),
			'pre_max' : FloatPar(1, 2, 'Duration (in seconds) of interval before the peak that is considered to find the start of the peak', '>0'),
			'post_max' : FloatPar(1, 2, 'Duration (in seconds) of interval after the peak that is considered to find the start of the peak', '>0')
			}
        return params


class DurationMax(_Indicator):
	"""
	Max duration of Peaks
	"""
	@classmethod
	def algorithm(cls, data, params):
		maxs, mins = _PeakDetection.get(data, delta=params['delta'])
        idxs_start, idxs_stop = _PeakSelection.get(data, maxs=maxs, pre_max = params['pre_max'], post_max = params['post_max'])
        
		if len(idxs_start) == 0:
			#WARNING
			return _np.nan
		else:
			fsamp = data.sampling_freq
			durations = []
			for I in range(len(idxs_start)):
				if (_np.isnan(idxs_stop[I]) == False) & (_np.isnan(idxs_start[I]) == False):
					durations.append( (idxs_stop[I] - idxs_start[I]) / fsamp ) # TODO: volendo si puo mettere in cache anche il calcolo di durations
				else:
					durations.append(_np.nan)

			return(_np.nanmax(_np.array(durations)))

	@classmethod
    def check_params(cls, params):
        params = {
			'delta': FloatPar(0, 2, 'Amplitude of the minimum peak', '>0'),
			'pre_max' : FloatPar(1, 2, 'Duration (in seconds) of interval before the peak that is considered to find the start of the peak', '>0'),
			'post_max' : FloatPar(1, 2, 'Duration (in seconds) of interval after the peak that is considered to find the start of the peak', '>0')
			}
        return params
	
	
class DurationMean(_Indicator):
	"""
    Mean duration of Peaks
    """
    @classmethod
    def algorithm(cls, data, params):
        maxs, mins = _PeakDetection.get(data, delta=params['delta'])
        idxs_start, idxs_stop = _PeakSelection.get(data, maxs=maxs, pre_max = params['pre_max'], post_max = params['post_max'])
        
        if len(idxs_start) == 0:
			#WARNING
			return _np.nan
		else:
			fsamp = data.sampling_freq
			durations = []
			for I in range(len(idxs_start)):
				if (_np.isnan(idxs_stop[I]) == False) & (_np.isnan(idxs_start[I]) == False):
					durations.append( (idxs_stop[I] - idxs_start[I]) / fsamp ) # TODO: volendo si puo mettere in cache anche il calcolo di durations
				else:
					durations.append(_np.nan)
			return(_np.nanmean(_np.array(durations)))

	@classmethod
    def check_params(cls, params):
        params = {
			'delta': FloatPar(0, 2, 'Amplitude of the minimum peak', '>0'),
			'pre_max' : FloatPar(1, 2, 'Duration (in seconds) of interval before the peak that is considered to find the start of the peak', '>0'),
			'post_max' : FloatPar(1, 2, 'Duration (in seconds) of interval after the peak that is considered to find the start of the peak', '>0')
			}
        return params


class SlopeMin(_Indicator):
	"""
    Min slope of Peaks
    """
    @classmethod
    def algorithm(cls, data, params):
        maxs, mins = _PeakDetection.get(data, delta=params['delta'])
        idxs_start, idxs_stop = _PeakSelection.get(data, maxs=maxs, pre_max = params['pre_max'], post_max = params['post_max'])
        
        if len(idxs_start) == 0:
			#WARNING
			return _np.nan
		else:
			fsamp = data.sampling_freq
			slopes = []
			for I in range(len(idxs_start)):
				if (_np.isnan(idxs_peak[I]) == False) & (_np.isnan(idxs_start[I]) == False):
					dy = data[idxs_peak[I]] - data[idxs_start[I]]
					dt = (idxs_peak[I] - idxs_start[I] ) / fsamp
					slopes.append(dy/dt) # TODO: volendo si puo mettere in cache anche il calcolo delle slopes
				else:
					slopes.append(_np.nan) # TODO: volendo si puo mettere in cache anche il calcolo delle slopes
			return(_np.nanmin(_np.array(slopes)))

	@classmethod
    def check_params(cls, params):
        params = {
			'delta': FloatPar(0, 2, 'Amplitude of the minimum peak', '>0'),
			'pre_max' : FloatPar(1, 2, 'Duration (in seconds) of interval before the peak that is considered to find the start of the peak', '>0'),
			'post_max' : FloatPar(1, 2, 'Duration (in seconds) of interval after the peak that is considered to find the start of the peak', '>0')
			}
        return params


class SlopeMax(_Indicator):
	"""
    Max slope of Peaks
    """
    @classmethod
    def algorithm(cls, data, params):
		maxs, mins = _PeakDetection.get(data, delta=params['delta'])
        idxs_start, idxs_stop = _PeakSelection.get(data, maxs=maxs, pre_max = params['pre_max'], post_max = params['post_max'])
        
        if len(idxs_start) == 0:
			#WARNING
			return _np.nan
		else:
			fsamp = data.sampling_freq
			slopes = []
			for I in range(len(idxs_start)):
				if (_np.isnan(idxs_peak[I]) == False) & (_np.isnan(idxs_start[I]) == False):
					dy = data[idxs_peak[I]] - data[idxs_start[I]]
					dt = (idxs_peak[I] - idxs_start[I] ) / fsamp
					slopes.append(dy/dt) # TODO: volendo si puo mettere in cache anche il calcolo delle slopes
				else:
					slopes.append(_np.nan) # TODO: volendo si puo mettere in cache anche il calcolo delle slopes
			return(_np.nanmax(_np.array(slopes)))

	@classmethod
    def check_params(cls, params):
        params = {
			'delta': FloatPar(0, 2, 'Amplitude of the minimum peak', '>0'),
			'pre_max' : FloatPar(1, 2, 'Duration (in seconds) of interval before the peak that is considered to find the start of the peak', '>0'),
			'post_max' : FloatPar(1, 2, 'Duration (in seconds) of interval after the peak that is considered to find the start of the peak', '>0')
			}
        return params


class SlopeMean(_Indicator):
	"""
    Min slope of Peaks
    """
    @classmethod
    def algorithm(cls, data, params):
		maxs, mins = _PeakDetection.get(data, delta=params['delta'])
        idxs_start, idxs_stop = _PeakSelection.get(data, maxs=maxs, pre_max = params['pre_max'], post_max = params['post_max'])
        
        if len(idxs_start) == 0:
			#WARNING
			return _np.nan
		else:
			fsamp = data.sampling_freq
			slopes = []
			for I in range(len(idxs_start)):
				if (_np.isnan(idxs_peak[I]) == False) & (_np.isnan(idxs_start[I]) == False):
					dy = data[idxs_peak[I]] - data[idxs_start[I]]
					dt = (idxs_peak[I] - idxs_start[I] ) / fsamp
					slopes.append(dy/dt) # TODO: volendo si puo mettere in cache anche il calcolo delle slopes
				else:
					slopes.append(_np.nan) # TODO: volendo si puo mettere in cache anche il calcolo delle slopes
			return(_np.nanmean(_np.array(slopes)))

	@classmethod
    def check_params(cls, params):
        params = {
			'delta': FloatPar(0, 2, 'Amplitude of the minimum peak', '>0'),
			'pre_max' : FloatPar(1, 2, 'Duration (in seconds) of interval before the peak that is considered to find the start of the peak', '>0'),
			'post_max' : FloatPar(1, 2, 'Duration (in seconds) of interval after the peak that is considered to find the start of the peak', '>0')
			}
        return params


