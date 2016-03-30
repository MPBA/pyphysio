# coding=utf-8
from __future__ import division
import numpy as _np
from Utility import abstractmethod as _abstract

__author__ = 'AleB'

# Everything in SECONDS (s) !!!


class _Signal(_np.ndarray):
    _MT_NATURE = "signal_nature"
    _MT_START_TIME = "start_time"
    _MT_META_DICT = "metadata"
    _MT_SAMPLING_FREQ = "sampling_freq"
    _MT_INFO_ATTR = "_pyphysio"

    def __new__(cls, y_values, sampling_freq, signal_nature="", start_time=0, meta=None):
        # noinspection PyNoneFunctionAssignment
        obj = _np.asarray(y_values).view(cls)
        obj._pyphysio = {
            cls._MT_NATURE: signal_nature,
            cls._MT_START_TIME: start_time,
            cls._MT_SAMPLING_FREQ: sampling_freq,
            cls._MT_META_DICT: meta if meta is not None else {}
        }
        return obj

    def __array_finalize__(self, obj):
        # __new__ called if obj is None
        if obj is not None and hasattr(obj, self._MT_INFO_ATTR):
            # The cache is not in MT_INFO_ATTR
            self._pyphysio = getattr(obj, self._MT_INFO_ATTR)

    def __array_wrap__(self, out_arr, context=None):
        # Just call the parent's
        # noinspection PyArgumentList
        return _np.ndarray.__array_wrap__(self, out_arr, context)

    @property
    def ph(self):
        return self._pyphysio

    @_abstract
    def get_duration(self):
        pass

    @_abstract
    def get_x_values(self, just_one=None):
        pass

    def get_y_values(self):
        return _np.asarray(self)

    def get_signal_nature(self):
        return self.ph[self._MT_NATURE]

    def get_sampling_freq(self):
        return self.ph[self._MT_SAMPLING_FREQ]

    def get_start_time(self):
        return self.ph[self._MT_START_TIME]

    def get_end_time(self):
        return self.get_start_time() + self.get_duration()

    def metadata(self):
        return self.ph[self._MT_META_DICT]

    def __repr__(self):
        return "<signal: " + self.get_signal_nature() + ", start_time: " + str(self.get_start_time()) + ">"


class EvenlySignal(_Signal):

    def get_duration(self):
        # Uses future division
        return len(self) / self.get_sampling_freq()

    def get_x_values(self, just_one=None):
        # Using future division
        if just_one is None:
            return _np.arange(self.get_start_time(), self.get_end_time(), 1. / self.get_sampling_freq())
        else:
            return self.get_start_time() + just_one / self.get_sampling_freq()

    # Works with timestamps (OLD)
    def getslice(self, f, l):
        # Using future division
        # find base_signal's indexes
        f = (f - self.get_start_time()) / self.get_sampling_freq()
        l = (l - self.get_start_time()) / self.get_sampling_freq()
        # clip the end
        # [:] has exclusive end
        if l > len(self):
            l = len(self)
        return EvenlySignal(self[f:l], self.get_sampling_freq(), self.get_signal_nature(), f)

    def __repr__(self):
        return _Signal.__repr__(self)[:-1] + " freq:" + str(self.get_sampling_freq()) + "Hz>\n" + self.view(
            _np.ndarray).__repr__()
    def resample(self, fout, kind='linear'):
	'''
	Resample a signal
	
	Parameters
	----------
	fout : float
	    The sampling frequency for resampling
	kind : str
	    Method for interpolation: 'linear', 'nearest', 'zero', 'slinear', 'quadratic, 'cubic'
	
	Returns
	-------
	resampled_signal : EvenlySignal
	    The resampled signal
	'''
	
	# TODO: check fout exists
	# TODO. check kind has correct value
	
	ratio = self.fsamp/fout
	
	if self.fsamp>=fout and ratio.is_integer(): #fast interpolation
	    indexes = np.arange(len(signal))
	    keep = (indexes%ratio == 0)
	    signal_out = signal[keep]
	    
	    self.sampling_freq = fout
	    # fix other signal properties
	    
	    return(signal_out)
	
	else:
	    indexes = np.arange(len(signal))        
	    indexes_out = np.arange(0, len(signal)-1+ratio, ratio) #TODO: check
	    if kind=='cubic':
		tck = interpolate.InterpolatedUnivariateSpline(indexes, signal)
	    else:
		tck = interpolate.interp1d(indexes, signal, kind=kind)
	    signal_out = tck(indexes_out)
	    
	    self.sampling_freq = fout
	    # fix other signal properties
	    
	    return(signal_out)


class _XYSignal(_Signal):
    _MT_X_VALUES = "x_values"

    def __new__(cls, y_values, x_values, sampling_freq, signal_nature, start_time, meta, check):
        assert not check or len(y_values) == len(x_values), \
            "Length mismatch (y:%d vs. x:%d)" % (len(y_values), len(x_values))
        obj = _Signal.__new__(cls, y_values, sampling_freq, signal_nature, start_time, meta)
        obj.ph[cls._MT_X_VALUES] = x_values
        return obj

    def get_x_values(self, just_one=None):
        if just_one is None:
            return self.ph[self._MT_X_VALUES]
        else:
            return self.ph[self._MT_X_VALUES][just_one]

    @_abstract
    def get_duration(self):
        pass

    @_abstract
    def getslice(self, f, l):
        pass

    def __repr__(self):
        return _Signal.__repr__(self) + "\ntimes-" + self.get_x_values().__repr__() + "\nvalues-" + self.view(
            _np.ndarray).__repr__()


class UnevenlySignal(_XYSignal):
    _MT_DURATION = "duration"

    def __new__(cls, y_values, time_values, duration, sampling_freq, signal_nature="", start_time=0, meta=None,
                check=True):
        obj = _XYSignal.__new__(cls, y_values, time_values, sampling_freq, signal_nature, start_time, meta, check)
        obj.ph[cls._MT_DURATION] = duration
        return obj

    def get_duration(self):
        return self.ph[self._MT_DURATION]

    # Works with timestamps
    def getslice(self, f, l):
        # find f & l indexes of indexes
        f = _np.searchsorted(self.get_x_values(), f)
        l = _np.searchsorted(self.get_x_values, l)
        return UnevenlySignal(self[f:l], self.get_x_values()[f:l], self.get_duration(), self.get_sampling_freq(),
                              self.get_signal_nature(), check=False)

    @classmethod
    def resample(self, kind='linear')
		"""
		Interpolate the UnevenlySignal to obtain an evenly spaced signal    
		
		Parameters
		----------
		kind : str
			Method for interpolation: 'linear', 'nearest', 'zero', 'slinear', 'quadratic, 'cubic'
		
		Returns
		-------
		interpolated_signal: nparray
			The interpolated signal
		"""
		data_y = _np.array(self.values())
		data_x = _np.array(self.indexes())
		total_len = 
		if self.indexes_array[0] != 0:
			data_x = _np.r[0, data_x]
			data_y = _np.r[self.input_array[0], self.input_array] #
		if self.indexes_array[-1] != total_len:
			self.indexes_array = _np.r[self.indexes_array, total_len-1]
			self.input_array = _np.r[self.input_array, self.input_array[-1]]
			
		# TODO: sistemare i tempi:
		# 1) indici originali
		# 2) fsamp originale
		# 3) lunghezza originale
		if kind=='cubic':
			tck = interpolate.InterpolatedUnivariateSpline(self.indexes_array, self.input_array)
		else:
			tck = interpolate.interp1d(self.indexes_array, self.input_array, kind=kind)
		sig_out = tck(_np.arange(total_len))
		
		sig_out = EvenlySignal(sig_out, ...)
		# fix other signal properties
		return(sig_out)

class UnevenlyTimeSignal(_XYSignal):

    def get_duration(self):
        return self.get_start_time() + self.get_x_values(len(self))

    # Works with timestamps
    def getslice(self, f, l):
        # find f & l indexes of indexes
        f = _np.searchsorted(self.get_x_values(), f)
        l = _np.searchsorted(self.get_x_values, l)
        return UnevenlySignal(self[f:l], self.get_x_values()[f:l], self.get_signal_nature(), check=False)


# Not used
# class EventsSignal(UnevenlyTimeSignal):
#     def __new__(cls, events, times, start_time=0, meta=None, check=True):
#         return UnevenlySignal(events, times, 0, 0, "events", start_time, meta, check)
#
#     # Works with timestamps
#     def getslice(self, f, l):
#         # find f & l indexes of indexes
#         f = _np.searchsorted(self.get_x_values(), f)
#         l = _np.searchsorted(self.get_x_values(), l)
#         return EventsSignal(self.get_x_values()[f:l], self.view(_np.ndarray)[f:l], check=False)
