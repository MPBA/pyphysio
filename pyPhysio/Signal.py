# coding=utf-8
from __future__ import division
import numpy as _np
from scipy import interpolate as _interp
from Utility import abstractmethod as _abstract, PhUI as _PhUI
from matplotlib.pyplot import plot as _plot

__author__ = 'AleB'

# Everything in SECONDS (s) !!!


class Signal(_np.ndarray):
    #FIXME: Make the following attributes "pickable"
    _MT_NATURE = "signal_nature"
    _MT_START_TIME = "start_time"
    _MT_START_INDEX = "start_index"
    _MT_SAMPLING_FREQ = "sampling_freq"
    _MT_INFO_ATTR = "_pyphysio"
    
    # TODO (Ale): check sui parametri del segnale: FSAMP > 0

    def __new__(cls, values, sampling_freq, signal_nature="", start_time=0):#, start_index=0):
        # noinspection PyNoneFunctionAssignment
        #TODO (feature) multichannel signals
        #TODO check values is 1-d
        obj = _np.asarray(_np.ravel(values)).view(cls)
        obj._pyphysio = {
            cls._MT_NATURE: signal_nature,
            cls._MT_START_TIME: start_time,
#            cls._MT_START_INDEX: start_index,
            cls._MT_SAMPLING_FREQ: sampling_freq,
        }
        return obj

    def __array_finalize__(self, obj):
        # __new__ called if obj is None
        if obj is not None and hasattr(obj, self._MT_INFO_ATTR):
            # The cache is not in MT_INFO_ATTR
            self._pyphysio = getattr(obj, self._MT_INFO_ATTR).copy()

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
    def get_times(self, just_one=None):
        pass

    def get_values(self):
        return _np.asarray(self)

    def get_signal_nature(self):
        return self.ph[self._MT_NATURE]

    def get_sampling_freq(self):
        return self.ph[self._MT_SAMPLING_FREQ]

    def get_start_time(self):
        return self.ph[self._MT_START_TIME]
        
    def get_end_time(self):
        return self.get_times()[-1]
    
    def plot(self, style=""):
        _plot(self.get_times(), self.get_values(), style)

    def __repr__(self):
        return "<signal: " + self.get_signal_nature() + ", start_time: " + str(self.get_start_time()) + ">"


class EvenlySignal(Signal):
    def get_duration(self):
        # Uses future division
        return len(self) / self.get_sampling_freq()

    def get_times(self):
        return _np.arange(len(self)) / self.get_sampling_freq() + self.get_start_time()

    def __repr__(self):
        return Signal.__repr__(self)[:-1] + " freq:" + str(self.get_sampling_freq()) + "Hz>\n" + self.view(
            _np.ndarray).__repr__()

    def resample(self, fout, kind='linear'):
        """
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
        """

        ratio = self.get_sampling_freq() / fout

        if fout < self.get_sampling_freq() and ratio.is_integer():  # fast interpolation
            signal_out = self.get_values()[::int(ratio)]
        else:
            # The last sample is doubled to allow the new size to be correct
            indexes = _np.arange(len(self) + 1)
            indexes_out = _np.arange(len(self) * fout / self.get_sampling_freq()) * ratio
            self_l = _np.append(self, self[-1])

            if kind == 'cubic':
                tck = _interp.InterpolatedUnivariateSpline(indexes, self_l)
            else:
                tck = _interp.interp1d(indexes, self_l, kind=kind)
            signal_out = tck(indexes_out)

        return EvenlySignal(signal_out, fout, self.get_signal_nature(), self.get_start_time())#, self.get_metadata())
        
    def segment_time(self, t_start, t_stop = None):
        """
        Segment the signal given a time interval

        Parameters
        ----------
        t_start : float
            The instant of the start of the interval
        t_stop : float 
            The instant of the end of the interval. By default is the end of the signal

        Returns
        -------
        portion : EvenlySignal
            The selected portion
        """
        
        #TODO: check
        signal_times = self.get_times()
        signal_values = self.get_values()
        
        if t_stop is None:
            t_stop = signal_times[-1]
        
        idx_start = _np.ceil((t_start - self.get_start_time()) * self.get_sampling_freq())
        idx_stop = _np.floor((t_stop  - self.get_start_time()) * self.get_sampling_freq())
        
        portion_values = signal_values[idx_start:idx_stop]
        t_0 = signal_times[idx_start]
        
        out_signal = EvenlySignal(portion_values, self.get_sampling_freq(), self.get_signal_nature(), t_0)#, self.get_metadata(), idx_start)
        
        return(out_signal)
    
    def segment_idx(self, idx_start, idx_stop):
        """
        Segment the signal given the indexes

        Parameters
        ----------
        idx_start : int
            The index of the start of the interval
        idx_stop : float 
            The index of the end of the interval. By default is the length of the signal 

        Returns
        -------
        portion : EvenlySignal
            The selected portion
        """
        #TODO: check
        signal_times = self.get_times()
        signal_values = self.get_values()
        
        if idx_stop is None:
            idx_stop = len(self)
            
        portion_values = signal_values[idx_start:idx_stop]
        t_0 = signal_times[idx_start]
        
        out_signal = EvenlySignal(portion_values, self.get_sampling_freq(), self.get_signal_nature(), t_0)#, self.get_metadata(), idx_start)
        
        return(out_signal)
        
    def __getslice__(self, i, j):
        o = Signal.__getslice__(self, i, j)
        if isinstance(o, Signal):
            o.ph[Signal._MT_START_INDEX] += i
        return o


class UnevenlySignal(Signal):
    _MT_X_VALUES = "x_values"
    _MT_ORIGINAL_LENGTH = "original_length"

    def __new__(cls, values, sampling_freq=1000, signal_nature="", start_time=None, indices = None, instants = None):#meta=None, start_index = 0, check=True):
        
        assert ((indices is not None) or (instants is not None)), "indices OR instants are required"
        
        assert not ((indices is not None) and (instants is not None)), "indices OR instants shold be given, not both"
        
        
        if indices is not None:
            assert len(values) == len(indices), "Length mismatch (y:%d vs. x:%d)" % (len(values), len(indices))
        
            indices = _np.array(indices)
            if start_time is None:
                start_time = 0
        else:
            assert len(values) == len(instants), "Length mismatch (y:%d vs. x:%d)" % (len(values), len(instants))
            
            if start_time is None:
                start_time = instants[0]
            indices = _np.round((instants - start_time)/sampling_freq).astype(int)
            
        obj = Signal.__new__(cls, values, sampling_freq, signal_nature, start_time)
        obj.ph[cls._MT_X_VALUES] = indices
        return obj

    def get_times(self):
        return (self.ph[self._MT_X_VALUES]) * self.get_sampling_freq() + self.get_start_time()

    def __repr__(self):
        return Signal.__repr__(self)[:-1] + " time resolution:" + str(1/self.get_sampling_freq()) + "s>\n" + self.view(
            _np.ndarray).__repr__() + "\times\n:" + self.get_times().__repr__()

    def __getslice__(self, i, j):
        o = Signal.__getslice__(self, i, j)
        if isinstance(o, UnevenlySignal):
            o.ph[UnevenlySignal._MT_X_VALUES] = o.ph[UnevenlySignal._MT_X_VALUES].__getslice__(i, j)
        return o

    def get_duration(self):
        return self.get_times()[-1] -  self.get_times()[0]

    def to_evenly(self, kind='cubic'):
        """
        Interpolate the UnevenlySignal to obtain an evenly spaced signal
        Parameters
        ----------
        kind : str
            Method for interpolation: 'linear', 'nearest', 'zero', 'slinear', 'quadratic, 'cubic'

        length : number
            Length in samples of the resulting signal. If not specified the last sample will be one after the last input point.

        Returns
        -------
        interpolated_signal: ndarray
            The interpolated signal
        """
        
        assert not (kind == 'cubic' and len(self)<=3), 'At least 4 samples needed for cubic interpolation' 

        data_x = self.ph[self._MT_X_VALUES]  # From a constant freq range

        data_y = self.get_values()

        # Cubic if needed
        if kind == 'cubic':
            tck = _interp.InterpolatedUnivariateSpline(data_x, data_y)
        else:
            tck = _interp.interp1d(data_x, data_y, kind=kind)
        
        x_out = _np.arange(data_x[0], data_x[-1]+1)
        sig_out = tck(x_out)

        # Init new signal
        sig_out = EvenlySignal(sig_out, self.get_sampling_freq(), self.get_signal_nature(), self.get_start_time())
        return sig_out
    
    def segment_time(self, t_start, t_stop = None):
        """
        Segment the signal given a time interval

        Parameters
        ----------
        t_start : float
            The instant of the start of the interval
        t_stop : float 
            The instant of the end of the interval. By default is the end of the signal

        Returns
        -------
        portion : UnvenlySignal
            The selected portion
        """
        
        #TODO: check
        signal_times = self.get_times()
        signal_values = self.get_values()
        
        if t_stop is None:
            t_stop = signal_times[-1]
        
        idx_start = _np.where(signal_times>=t_start)[0][0]
        idx_stop = _np.where(signal_times<=t_stop)[0][-1]
        
        portion_values = signal_values[idx_start:idx_stop+1]
        portion_times = signal_times[idx_start:idx_stop+1]
                
        t_0 = signal_times[idx_start]
        
        out_signal = UnevenlySignal(portion_values, self.get_sampling_freq(), self.get_signal_nature(), t_0, instants = portion_times)
        
        return(out_signal)
    
    def segment_idx(self, idx_start, idx_stop):
        """
        Segment the signal given the indexes

        Parameters
        ----------
        idx_start : int
            The index of the start of the interval
        idx_stop : float 
            The index of the end of the interval. By default is the length of the signal 

        Returns
        -------
        portion : EvenlySignal
            The selected portion
        """
        #TODO: check
        signal_values = self.get_values()
        signal_indices = self.ph[self._MT_X_VALUES]
        
        if idx_stop is None:
            idx_stop = len(self)
        
        i_start = _np.where(signal_indices>=idx_start)[0][0]
        i_stop = _np.where(signal_indices<=idx_stop)[0][-1]
        
        portion_values = signal_values[i_start:i_stop]
        portion_indices = signal_indices[i_start:i_stop]
        
        t_0 = self.get_times()[i_start]
        
        out_signal = UnevenlySignal(portion_values, self.get_sampling_freq(), self.get_signal_nature(), t_0, indices = portion_indices)
        
        return(out_signal)
        

#class EventsSignal(UnevenlySignal):
#    def __new__(cls, values, times, orig_sampling_freq=1, orig_length=None, signal_nature="", start_time=0,
#                meta=None, check=True):
#        return UnevenlySignal.__new__(cls, values, times * orig_sampling_freq, orig_sampling_freq, orig_length, signal_nature, start_time,
#                                      meta, check)