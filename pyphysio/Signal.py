# coding=utf-8
from __future__ import division
import numpy as _np
from scipy import interpolate as _interp
from matplotlib.pyplot import plot as _plot, vlines as _vlines, xlabel as _xlabel, ylabel as _ylabel, grid as _grid 
from matplotlib.pyplot import subplot as _subplot, tight_layout as _tight_layout, subplots_adjust as _subplots_adjust, xlim as _xlim
from numbers import Number as _Number
from pyphysio.Utility import abstractmethod as _abstract, PhUI as _PhUI
import copy
#from pyphysio.filters.Filters import ImputeNAN as _ImputeNAN
__author__ = 'AleB'


# TODO: Consider collapsing classes

def from_pickleable(pickle):
    """
    Builds a Signal using the pickleable tuple version of it.
    :param pickle: Tuple of the form (Signal, ph dict).
    :return: Signal
    """
    d, ph = pickle
    assert isinstance(d, Signal)
    assert isinstance(ph, dict)
    d._pyphysio = ph
    return d


def from_pickle(path):
    """
    Loads a Signal from a pickle file given the path.
    :param path: File system path to the pickle file.
    :return: A Signal.
    """
    from gzip import open
    from pickle import load
    f = open(path)
    p = load(f)
    f.close()
    return from_pickleable(p)


class Signal(_np.ndarray):
    _MT_NATURE = "signal_type"
    _MT_START_TIME = "start_time"
    _MT_SAMPLING_FREQ = "sampling_freq"
    _MT_INFO_ATTR = "_pyphysio"

    def __new__(cls, values, sampling_freq, start_time=None, signal_type=""):
        assert sampling_freq > 0, "The sampling frequency cannot be zero or negative"
        assert start_time is None or isinstance(start_time, _Number), "Start time is not numeric"
        obj = _np.asarray(values).view(cls)
                    
        if len(obj) == 0:
            _PhUI.i("Creating empty " + cls.__name__)
            
        obj._pyphysio = {
            cls._MT_SAMPLING_FREQ: sampling_freq,
            cls._MT_START_TIME: start_time if start_time is not None else 0,
            cls._MT_NATURE: signal_type
        }
        setattr(obj, "_mutated", False)
        return obj

    def __array_finalize__(self, obj):
        # __new__ called if obj is None
        if obj is not None and hasattr(obj, self._MT_INFO_ATTR):
            # The cache is not in MT_INFO_ATTR
            self._pyphysio = getattr(obj, self._MT_INFO_ATTR).copy()

    def __array_wrap__(self, out_arr, context=None):
        # Just call the parent's
        # noinspection PyArgumentList
        if isinstance(out_arr, Signal):
            return _np.ndarray.__array_wrap__(self, out_arr, context)
        else:
            return out_arr

    @property
    def ph(self):
        return self._pyphysio

    def clone(self):
        obj = self.copy()
        obj._pyphysio = copy.deepcopy(self.ph)
        return(obj)
    
    def is_multi(self):
        return(self.get_nchannels()>1)
    
    def get_values(self):
        return _np.asarray(self)
    
    def get_nchannels(self):
        if self.ndim>1:
            return(self.shape[1])
        else:
            return(1)
        
    def get_sampling_freq(self):
        return self.ph[self._MT_SAMPLING_FREQ]

    def set_sampling_freq(self, value):
        setattr(self, "_mutated", True)
        self.ph[self._MT_SAMPLING_FREQ] = value
    
    def get_start_time(self):
        return self.ph[self._MT_START_TIME]

    def set_start_time(self, value):
        setattr(self, "_mutated", True)
        self.ph[self._MT_START_TIME] = value    
    
    def get_signal_type(self):
        return self.ph[self._MT_NATURE]

    def set_signal_type(self, value):
        setattr(self, "_mutated", True)
        self.ph[self._MT_NATURE] = value
    
    def get_duration(self):
        return self.get_end_time() - self.get_start_time()
    
    def get_idx(self, time):
        idx = int((time - self.get_start_time()) * self.get_sampling_freq())
        if idx < 0:
            idx=0
        return(idx)
        
    @_abstract
    def clone_properties(self):
        pass
    
    @_abstract
    def get_times(self):
        pass
    
    @_abstract
    def get_end_time(self):
        pass

    @_abstract
    def get_iidx(self, time):
        pass

    @_abstract
    def get_time(self, idx):
        pass

    @_abstract
    def get_time_from_iidx(self, iidx):
        pass

    @_abstract
    def resample(self, fout, kind='linear'):
        pass

    @_abstract
    def segment_idx(self, t_start, t_stop=None):
        pass

    @_abstract
    def segment_iidx(self, t_start, t_stop=None):
        pass

    @_abstract
    def segment_time(self, t_start, t_stop=None):
        pass

    def plot(self, style="", vlines_height=1000):
        _xlabel("time")
        _ylabel(self.get_signal_type())
        _grid()
        if len(style) > 0 and style[0] == "|":
            return _vlines(self.get_times(), -vlines_height / 2, vlines_height / 2, style[1:])
        else:
            return _plot(self.get_times(), self.get_values(), style)

    @property
    def pickleable(self):
        """
        Returns a pickleable tuple of this Signal.
        :return: Tuple (Signal, ph dict).
        """
        return self, self.ph

    def to_pickle(self, path):
        """
        Saves this Signal into a pickle file.
        :param path: File system path to the file to write (create/overwrite).
        """
        from gzip import open
        from pickle import dump
        f = open(path, "wb")
        dump(self.pickleable, f, protocol=2)
        f.close()

#    def impute_nans(self):
#        self = ImputeNAN()(self)
        
    def __repr__(self):
        return "<signal: " + self.get_signal_type() + ", start_time: " + str(self.get_start_time()) + ">"

    def __getslice__(self, i, j):
        return self.segment_iidx(i, j)


class EvenlySignal(Signal):
    """
    Evenly spaced signal
    
    Attributes:
    -----------
    
    data : numpy.array
        Values of the signal
    sampling_freq : float, >0
        Sampling frequency
    start_time: float,
        Instant of signal start
    signal_type : str, default = ''
        Type of signal (e.g. 'ECG', 'EDA')
    """

    def clone_properties(self, new_values):
        x_new = EvenlySignal(new_values,
                             self.get_sampling_freq(),
                             self.get_start_time(),
                             self.get_signal_type())
        return(x_new)

    def get_times(self):
        return _np.arange(len(self)) / self.get_sampling_freq() + self.get_start_time()

    def get_end_time(self):
        return self.get_time(len(self) - 1) + 1. / self.get_sampling_freq()

    def get_iidx(self, time):
        return self.get_idx(time)

    def get_time(self, idx):
        return idx / self.get_sampling_freq() + self.get_start_time() if idx is not None else None

    def get_time_from_iidx(self, iidx):
        return self.get_time(iidx)

    def get_value_t(self, instant):
        values = self.get_values()
        nearest_idx = int(_np.round(self.get_sampling_freq() * (instant - self.get_start_time())))
        assert nearest_idx < len(self), "Required instant is after the end of the signal"  # return self[-1]
        assert nearest_idx >= 0, "Required instant is before the start of the signal"  # return self[0]
        
        return values[nearest_idx]
    
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

        return EvenlySignal(values=signal_out,
                            sampling_freq=fout,
                            signal_type=self.get_signal_type(),
                            start_time=self.get_start_time())

    # TRYME
    def segment_idx(self, idx_start, idx_stop=None):
        """
        Segment the signal given the indexes

        Parameters
        ----------
        idx_start : int or None
            The index of the start of the interval
        idx_stop : int or None
            The index of the end of the interval. By default is the length of the signal

        Returns
        -------
        portion : EvenlySignal
            The selected portion
        """
        return self.segment_iidx(idx_start, idx_stop)

    # TRYME
    def segment_iidx(self, iidx_start, iidx_stop=None):

        signal_values = self.get_values()

        if iidx_start is None:
            iidx_start = 0
        if iidx_stop is None:
            iidx_stop = len(self)

        values = signal_values[int(iidx_start):int(iidx_stop)]

        out_signal = self.clone_properties(values)
        out_signal.set_start_time(self.get_time(iidx_start))
        return out_signal

    # TRYME
    def segment_time(self, t_start, t_stop=None):
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

        return self.segment_idx(self.get_idx(t_start), self.get_idx(t_stop))

    def to_csv(self, filename, comment=''):
        values = self.get_values()
        times = self.get_times()
        header = self.get_signal_type() + ' \n' + 'Fsamp: ' + str(
            self.get_sampling_freq()) + '\n' + comment + '\nidx,time,value'

        _np.savetxt(filename, _np.c_[times, values], delimiter=',', header=header, comments='')

    def __repr__(self):
        return Signal.__repr__(self)[:-1] + " freq:" + str(self.get_sampling_freq()) + "Hz>\n" + self.view(
            _np.ndarray).__repr__()


class UnevenlySignal(Signal):
    """
    Unevenly spaced signal
    
    Attributes:
    -----------
    
    data : numpy.array
        Values of the signal
    sampling_freq : float, >0
        Sampling frequency
    start_time: float,
        Instant of signal start
    signal_type : str, default = ''
        Type of signal (e.g. 'ECG', 'EDA')
    
    
    x_values : numpy.array of int
        Instants, or indices when the values are measured.
    x_type : str
        Type of x values given.
        Can be 'indices' or 'instants'

    duration: float,
        Duration of the original EvenlySignal, if any. Duration is needed to have information about the duration of the
        last sample, if None the last sample will last 1. / fsamp.
    """

    _MT_X_INDICES = "x_values"
    _MT_DURATION = "duration"

    def __new__(cls, values, sampling_freq=1000, start_time=None, signal_type="", x_values=None, x_type='instants',
                duration=None):
        assert x_values is not None, "x_values are missing"
        assert x_type in ['indices', 'instants'], "x_type not in ['indices', 'instants']"
        x_values = _np.asarray(x_values)
        assert len(x_values) == len(values), "Length mismatch (y:%d vs. x:%d)" % (len(values), len(x_values))
        assert len(_np.where(_np.diff(x_values) <= 0)[0]) == 0, 'Given x_values are not strictly monotonic'

        
        if x_type == 'indices':
            # Keep indices, set start_time
            if start_time is None:
                start_time = 0
        else:
            
            # Get indices removing start_time
            if start_time is None:
                start_time = x_values[0]
            else:
                assert start_time <= x_values[0], "More than one sample at or before start_time"
            
            # WARN: limitation to 10 decimals due to workaround to prevent wrong cast flooring
            # (e.g. np.floor(0.29 * 100) == 28)
            x_values = _np.round((x_values - start_time) * sampling_freq, 10).astype(int)

        # adding 1/f cause end_time is exclusive 
        # Doesn't work when I put the last sample of a signal (we should add 1/fsamp to Signal.get_duration() too)
        min_duration = (x_values[-1] + 1.) / sampling_freq if len(x_values) > 0 else 0
        assert duration is None or duration >= min_duration, \
            "The specified duration is less than the one of the x_values"

        obj = Signal.__new__(cls, values=values,
                             sampling_freq=sampling_freq,
                             start_time=start_time,
                             signal_type=signal_type)

        if duration is None:
            duration = min_duration

        obj.ph[cls._MT_X_INDICES] = x_values
        obj.ph[cls._MT_DURATION] = duration
        return obj

    def clone_properties(self, new_values, new_x, new_x_type):
        x_new = UnevenlySignal(new_values,
                               self.get_sampling_freq(),
                               self.get_start_time(),
                               self.get_signal_type(),
                               new_x,
                               new_x_type)
        # TODO: test clone properties
        return(x_new)

    def get_duration(self):
        return self.ph[UnevenlySignal._MT_DURATION]

    def get_end_time(self):
        return self.get_start_time() + self.get_duration()

    def get_times(self):
        return self.ph[self._MT_X_INDICES] / self.get_sampling_freq() + self.get_start_time()

    def get_indices(self):
        return self.ph[self._MT_X_INDICES]

    def get_time(self, idx):
        return idx / self.get_sampling_freq() + self.get_start_time() if idx is not None else None

    def get_time_from_iidx(self, iidx):
        if len(self) == 0:
            return self.get_start_time()
        elif int(iidx) < len(self):
            return self.get_indices()[int(iidx)] / self.get_sampling_freq() + self.get_start_time()
        else:
            return self.get_time_from_iidx(-1)

    def get_iidx(self, time):
        return self.get_iidx_from_idx((time - self.get_start_time()) * self.get_sampling_freq())

    def get_iidx_from_idx(self, idx):
        if idx >= self.get_indices()[0]:
            return int(_np.searchsorted(self.get_indices(), idx))
        else:
            return None

    def to_csv(self, filename, comment=''):
        values = self.get_values()
        times = self.get_times()
        idxs = self.get_indices()
        header = self.get_signal_type() + ' \n' + 'Fsamp: ' + str(
            self.get_sampling_freq()) + '\n' + comment + '\nidx,time,value'

        _np.savetxt(filename, _np.c_[idxs, times, values], delimiter=',', header=header, comments='')

    def to_evenly(self, kind='cubic'):
        """
        Interpolate the UnevenlySignal to obtain an evenly spaced signal
        Parameters
        ----------
        kind : str
            Method for interpolation: 'linear', 'nearest', 'zero', 'slinear', 'quadratic, 'cubic'

        Returns
        -------
        interpolated_signal: ndarray
            The interpolated signal
        """

        assert kind != 'cubic' or len(self) > 3, "At least 4 samples needed for cubic interpolation"

        data_x = self.ph[self._MT_X_INDICES]  # From a constant freq range
        data_y = self.get_values()

        # Cubic if needed
        if kind == 'cubic':
            tck = _interp.InterpolatedUnivariateSpline(data_x, data_y)
        else:
            tck = _interp.interp1d(data_x, data_y, kind=kind)

        # Exclusive end, same x_value
        x_out = _np.arange(data_x[0], data_x[-1] + 1)
        sig_out = tck(x_out)

        # Init new signal
        sig_out = EvenlySignal(values=sig_out,
                               sampling_freq=self.get_sampling_freq(),
                               signal_type=self.get_signal_type(),
                               start_time=self.get_time_from_iidx(0))

        return sig_out

    def resample(self, fout, kind='linear'):
        return self.to_evenly(kind).resample(fout, kind)

    def segment_time(self, t_start, t_stop=None):
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

        return self.segment_idx(self.get_idx(t_start) if t_start is not None else None,
                                self.get_idx(t_stop) if t_stop is not None else None)

    def segment_idx(self, idx_start, idx_stop=None):
        """
        Segment the signal given the indexes

        Parameters
        ----------
        idx_start : int
            The index of the start of the interval
        idx_stop : float
            The index of the end of the interval. By default is the end of the signal

        Returns
        -------
        portion : UnvenlySignal
            The selected portion
        """
        if idx_start is None:
            idx_start = 0
        if idx_stop is None:
            idx_stop = self.get_indices()[-1]

        iib = self.get_iidx_from_idx(idx_start)
        iie = self.get_iidx_from_idx(idx_stop)

        if iib is None and iie is None:
            iidx_start = iidx_stop = idx_start = idx_stop = 0
        else:
            iidx_start = int(iib) if iib is not None else 0
            iidx_stop = int(iie) if iie is not None else -1

        return UnevenlySignal(values=self.get_values()[iidx_start:iidx_stop],
                              x_values=self.get_indices()[iidx_start:iidx_stop] - idx_start,
                              sampling_freq=self.get_sampling_freq(),
                              signal_type=self.get_signal_type(),
                              start_time=self.get_time(idx_start),
                              x_type='indices',
                              duration=(idx_stop - idx_start) / self.get_sampling_freq())

    def segment_iidx(self, iidx_start, iidx_stop=None):
        """
        Segment the signal given the inner indexes

        Parameters
        ----------
        iidx_start : int
            The index of the start of the interval
        iidx_stop : float
            The index of the end of the interval. By default is the end of the signal

        Returns
        -------
        portion : UnvenlySignal
            The selected portion
        """
        if iidx_stop is None:
            iidx_stop = len(self)
        if iidx_start is None:
            iidx_start = 0
        if iidx_stop < len(self):
            idx_stop = self.get_indices()[int(iidx_stop)]
        else:
            idx_stop = self.get_indices()[-1] + 1
        idx_start = self.get_indices()[int(iidx_start)]

        return UnevenlySignal(values=self.get_values()[int(iidx_start):int(iidx_stop)],
                              x_values=self.get_indices()[int(iidx_start):int(iidx_stop)]
                              - self.get_indices()[int(iidx_start)],
                              sampling_freq=self.get_sampling_freq(),
                              signal_type=self.get_signal_type(),
                              start_time=self.get_time_from_iidx(iidx_start),
                              x_type='indices',
                              duration=(idx_stop - idx_start) / self.get_sampling_freq())

    def __repr__(self):
        return Signal.__repr__(self)[:-1] + " time resolution:" + str(1 / self.get_sampling_freq()) + "s>\n" + \
               self.get_values().__repr__() + " Times\n:" + self.get_times().__repr__()


class MultiEvenly(EvenlySignal):
    
    def __new__(cls, values, sampling_freq, start_time=None, signal_type='raw', info = {}):
        assert sampling_freq > 0, "The sampling frequency cannot be zero or negative"
        assert start_time is None or isinstance(start_time, _Number), "Start time is not numeric"
        obj = Signal.__new__(cls, values=values, sampling_freq=sampling_freq, start_time=start_time, signal_type=signal_type)
        
        return obj

    def clone_properties(self, new_values):
        x_new = MultiEvenly(new_values,
                            self.get_sampling_freq(),
                            self.get_start_time(),
                            self.get_signal_type())
        return(x_new)
    
    
    def set_start_time(self, value):
        setattr(self, "_mutated", True)
        self.ph[self._MT_START_TIME] = value  
        stim = self.get_stim()
        stim.set_start_time(value)
        
    def get_channel(self, i_ch):
        ch_values = self.get_values()[:,i_ch]
        return(EvenlySignal(ch_values, self.get_sampling_freq(), self.get_start_time(), self.get_signal_type()))
        
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
            values_out = self.get_values()[::int(ratio),:]
        else:
            indexes = _np.arange(len(self) + 1)
            indexes_out = _np.arange(len(self) * fout / self.get_sampling_freq()) * ratio
            
            values = self.get_values()
            values = _np.vstack([values, values[-1,:]])
            tck = _interp.interp1d(indexes, values, kind=kind, axis=0)
            
            values_out = tck(indexes_out)
    
        signal_out = self.clone_properties(values_out)
        signal_out.set_sampling_freq(fout)
        return(signal_out)
        

    # TRYME
    def segment_iidx(self, iidx_start, iidx_stop=None):

        signal_values = self.get_values()

        if iidx_start is None:
            iidx_start = 0
        if iidx_stop is None:
            iidx_stop = len(self)

        values = signal_values[int(iidx_start):int(iidx_stop),:]

        out_signal = self.clone_properties(values)
        out_stim = self.get_stim().clone()
        out_stim = out_stim.segment_iidx(iidx_start, iidx_stop)
        out_signal.set_stim(out_stim)
        
        out_signal.set_start_time(self.get_time(iidx_start))
        return out_signal
    
    
    def plot(self, style=""):
        _grid()
        n_ch = self.get_nchannels()
    
        t_start = self.get_start_time()
        
        n_rows = int(_np.ceil(n_ch/4))
        ax1 = _subplot(n_rows, 4, 1)
        for i_ch in range(n_ch):
            _subplot(n_rows, 4, i_ch+1, sharex=ax1)
            self.get_channel(i_ch).plot(style)
            _ylabel(i_ch)
            _vlines(t_start, _np.nanmin(self.get_channel(i_ch)), _np.nanmax(self.get_channel(i_ch)), 'k')
            _grid()
        _xlim(self.get_start_time(), self.get_end_time())
        _tight_layout()
        _subplots_adjust(top=0.9, bottom=0.01, left=0.05, right=0.95, hspace=0.3, wspace=0.25)

    def to_csv(self, filename, comment='', fmt = '%.6f'): 
        values = self.get_values()
        times = self.get_times()
        header = self.get_signal_type() + ' \n' + 'Fsamp: ' + str(self.get_sampling_freq()) + '\n' + comment + '\nidx,time'+''.join([f',ch{x}' for x in range(self.get_nchannels())])
        _np.savetxt(filename, _np.c_[_np.arange(len(times)), times, values], delimiter=',', header=header, comments='', fmt = fmt)

    def __repr__(self):
        return f"<start_time: {self.get_start_time()}> freq:  {self.get_sampling_freq()} Hz> {self.view(_np.ndarray).__repr__()}"

    def __getslice__(self, i, j):
        return self.segment_iidx(i, j)