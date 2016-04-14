# coding=utf-8
from __future__ import division
import numpy as _np
from scipy import interpolate as _interp
from Utility import abstractmethod as _abstract, AbstractCalledError as _ACError

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

    def get_metadata(self):
        return self.ph[self._MT_META_DICT]

    def __repr__(self):
        return "<signal: " + self.get_signal_nature() + ", start_time: " + str(self.get_start_time()) + ">"


class EvenlySignal(_Signal):

    def get_duration(self):
        # Uses future division
<<<<<<< HEAD
        return len(self) / self.sampling_freq
    
    @property
    def nsamples(self):
        # Uses future division
        return len(self)
=======
        return len(self) / self.get_sampling_freq()
>>>>>>> aa01c96c835c133c1c60ed9fa1a23dca6afbdeac

    def get_x_values(self, just_one=None):
        # Using future division
        if just_one is None:
            return _np.arange(0, self.get_duration(), 1. / self.get_sampling_freq())
        else:
            return just_one / self.get_sampling_freq()

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
<<<<<<< HEAD
        '''
        Resample a signal
        
=======
        """
        Resample a signal

>>>>>>> aa01c96c835c133c1c60ed9fa1a23dca6afbdeac
        Parameters
        ----------
        fout : float
            The sampling frequency for resampling
        kind : str
            Method for interpolation: 'linear', 'nearest', 'zero', 'slinear', 'quadratic, 'cubic'
<<<<<<< HEAD
        
=======

>>>>>>> aa01c96c835c133c1c60ed9fa1a23dca6afbdeac
        Returns
        -------
        resampled_signal : EvenlySignal
            The resampled signal
<<<<<<< HEAD
        '''
        # TODO: check fout exists
        # TODO. check kind has correct value
        
        ratio = self.sampling_freq/fout
        
        if self.sampling_freq>=fout and ratio.is_integer(): #fast interpolation
            indexes = _np.arange(self.nsamples)
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

class UnevenlySignal(Signal):
    _MT_TIMES = "times"

    def __new__(cls, input_array, indexes_array, sampling_freq, length_original, signal_nature="", start_time=0, meta=None, check=True): # FIX 
        # TODO check: useful O(n) monotonicity check?
        assert not check or len(input_array) == len(times_array),\
            "Length mismatch (%d vs. %d)" % (len(input_array), len(times_array))
        assert all(i > 0 for i in _np.diff(times_array)), "Time is not monotonic"
        obj = Signal(input_array, signal_nature, start_time, meta).view(cls)
        obj.ph[cls._MT_TIMES] = times_array
=======
        """

        ratio = self.get_sampling_freq()/fout

        if self.get_sampling_freq() >= fout and ratio.is_integer():  # fast interpolation
            indexes = _np.arange(len(self))
            keep = indexes % ratio == 0
            signal_out = self[keep]
        else:
            indexes = _np.arange(len(self))
            indexes_out = _np.arange(0, len(self)-1+ratio, ratio)  # TODO: check
            if kind == 'cubic':
                tck = _interp.InterpolatedUnivariateSpline(indexes, self)
            else:
                tck = _interp.interp1d(indexes, self, kind=kind)
            signal_out = tck(indexes_out)

        return EvenlySignal(signal_out, fout, self.get_signal_nature(), self.get_start_time(), self.get_metadata())


class _XYSignal(_Signal):
    _MT_X_VALUES = "x_values"

    def __new__(cls, y_values, x_values, sampling_freq, signal_nature, start_time, meta, check):
        assert not check or len(y_values) == len(x_values), \
            "Length mismatch (y:%d vs. x:%d)" % (len(y_values), len(x_values))
        obj = _Signal.__new__(cls, y_values, sampling_freq, signal_nature, start_time, meta)
        obj.ph[cls._MT_X_VALUES] = x_values
>>>>>>> aa01c96c835c133c1c60ed9fa1a23dca6afbdeac
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
    def to_evenly(self, kind='linear'):
        pass

    @_abstract
    def getslice(self, f, l):
<<<<<<< HEAD
        # find f & l indexes of indexes
        f = _np.searchsorted(self.get_times(), f)
        l = _np.searchsorted(self.get_times(), l)
        return UnevenlySignal(self[f:l], self.get_times()[f:l], self.signal_nature, check=False)

    @classmethod
    def resample(self, kind='linear')
        """
        Interpolate the UnevenlySignal to obtain an evenly spaced signal    
        
=======
        pass

    def __repr__(self):
        return _Signal.__repr__(self) + "\ny-values\n" + self.view(_np.ndarray).__repr__() +\
            "\nx-times\n" + self.get_x_values().__repr__()

    def _to_evenly(self, kind='linear', length=None):
        """
        Interpolate the UnevenlySignal to obtain an evenly spaced signal

>>>>>>> aa01c96c835c133c1c60ed9fa1a23dca6afbdeac
        Parameters
        ----------
        kind : str
            Method for interpolation: 'linear', 'nearest', 'zero', 'slinear', 'quadratic, 'cubic'
<<<<<<< HEAD
        
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

class EventsSignal(UnevenlySignal):
    def __new__(cls, events, times, meta=None, checks=True):
        return UnevenlySignal(events, times, "events", meta, checks)
=======

        length : number
            Length of the resulting signal. If not specified the last sample will be one after the last input point.

        Returns
        -------
        interpolated_signal: ndarray
            The interpolated signal
        """

        data_x = self.get_x_values()
        data_y = self.get_y_values()

        # Add padding
        if self.get_x_values(0) != 0:
            data_x = _np.r_[0, data_x]
            data_y = _np.r_[data_y[0], data_y]
        if self.get_x_values(-1) != length - 1:
            data_x = _np.r_[data_x, length - 1]
            data_y = _np.r_[data_y, data_y[-1]]

        # Cubic if needed
        if kind == 'cubic':
            tck = _interp.InterpolatedUnivariateSpline(data_x, data_y)
        else:
            tck = _interp.interp1d(data_x, data_y, kind=kind)
        sig_out = tck(_np.arange(length))

        # Init new signal
        sig_out = EvenlySignal(sig_out, self.get_sampling_freq(), self.get_signal_nature(), self.get_start_time(),
                               self.get_metadata())
        return sig_out


class UnevenlySignal(_XYSignal):
    _MT_ORIGINAL_LENGTH = "duration"

    def __new__(cls, y_values, indexes, original_length, sampling_freq, signal_nature="", start_time=0, meta=None,
                check=True):
        obj = _XYSignal.__new__(cls, y_values, indexes, sampling_freq, signal_nature, start_time, meta, check)
        obj.ph[cls._MT_ORIGINAL_LENGTH] = original_length
        return obj

    def get_duration(self):
        return self.ph[self._MT_ORIGINAL_LENGTH] / self.get_sampling_freq()

    def get_original_length(self):
        return self.ph[self._MT_ORIGINAL_LENGTH]
>>>>>>> aa01c96c835c133c1c60ed9fa1a23dca6afbdeac

    # Works with timestamps
    def getslice(self, f, l):
        # find f & l indexes of indexes
        f = _np.searchsorted(self.get_x_values(), f)
        l = _np.searchsorted(self.get_x_values, l)
        return UnevenlySignal(self[f:l], self.get_x_values()[f:l], self.get_duration(), self.get_sampling_freq(),
                              self.get_signal_nature(), check=False)

    def to_evenly(self, kind='linear'):
        return self._to_evenly(kind, self.get_original_length())


class UnevenlyTimeSignal(_XYSignal):

    def get_duration(self):
        return self.get_x_values(-1)

    def to_evenly(self, kind='linear'):
        return self._to_evenly(kind, self.get_duration() * self.get_sampling_freq())

    # # Works with timestamps
    # def getslice(self, f, l):
    #     # find f & l indexes of indexes
    #     f = _np.searchsorted(self.get_x_values(), f)
    #     l = _np.searchsorted(self.get_x_values(), l)
    #     return UnevenlySignal(self[f:l], self.get_x_values()[f:l], self.get_signal_nature())


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
