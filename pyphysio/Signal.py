# coding=utf-8
from __future__ import division
import numpy as _np
from scipy import interpolate as _interp
from pyphysio.Utility import abstractmethod as _abstract
from matplotlib.pyplot import plot as _plot, vlines as _vlines

__author__ = 'AleB'

# TODO: Consider collapsing classes


class Signal(_np.ndarray):
    # TODO: Make the following attributes "pickleable"
    _MT_NATURE = "signal_nature"
    _MT_START_TIME = "start_time"
    _MT_SAMPLING_FREQ = "sampling_freq"
    _MT_INFO_ATTR = "_pyphysio"

    def __new__(cls, values, sampling_freq, start_time=None, signal_nature=""):
        # TODO (feature) multichannel signals
        assert sampling_freq > 0, "The sampling frequency cannot be zero or negative"
        assert not isinstance(start_time, str), "Start time is a String"
        obj = _np.asarray(values).view(cls)
        assert obj.ndim == 1, "Dimension not 1"
        obj._pyphysio = {
            cls._MT_NATURE: signal_nature,
            cls._MT_START_TIME: start_time if start_time is not None else 0,
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

    def get_duration(self):
        return self.get_end_time() - self.get_start_time()

    @_abstract
    def get_times(self):
        pass

    def get_values(self):
        return _np.asarray(self)

    def get_signal_nature(self):
        return self.ph[self._MT_NATURE]

    def set_signal_nature(self, value):
        self.ph[self._MT_NATURE] = value

    def get_sampling_freq(self):
        return self.ph[self._MT_SAMPLING_FREQ]

    def set_sampling_freq(self, value):
        self.ph[self._MT_SAMPLING_FREQ] = value

    def get_start_time(self):
        return self.ph[self._MT_START_TIME]

    def set_start_time(self, value):
        self.ph[self._MT_START_TIME] = value

    @_abstract
    def get_end_time(self):
        pass

    def get_idx(self, time):
        return (time - self.get_start_time()) * self.get_sampling_freq() if time < self.get_end_time() else None

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
        if len(style) > 0 and style[0] == "|":
            return _vlines(self.get_times(), -vlines_height/2, vlines_height/2, style[1:])
        else:
            return _plot(self.get_times(), self.get_values(), style)

    def __repr__(self):
        return "<signal: " + self.get_signal_nature() + ", start_time: " + str(self.get_start_time()) + ">"

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
    signal_nature : str, default = ''
        Type of signal (e.g. 'ECG', 'EDA')
    start_time: float,
        Instant of signal start
    """

    def get_times(self):
        return _np.arange(len(self)) / self.get_sampling_freq() + self.get_start_time()

    def get_end_time(self):
        return self.get_time(len(self) - 1) + 1. / self.get_sampling_freq()

    def get_time(self, idx):
        return idx / self.get_sampling_freq() + self.get_start_time()\
            if idx is not None and idx < len(self) else None

    def get_iidx(self, time):
        return self.get_idx(time)

    def get_time_from_iidx(self, iidx):
        return self.get_time(iidx)

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
                            signal_nature=self.get_signal_nature(),
                            start_time=self.get_start_time())

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

    # TRYME
    def segment_idx(self, idx_start, idx_stop=None):
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
        return self.segment_iidx(idx_start, idx_stop)

    # TRYME
    def segment_iidx(self, iidx_start, iidx_stop=None):

        signal_values = self.get_values()

        if iidx_start is None:
            iidx_start = 0
        if iidx_stop is None:
            iidx_stop = len(self)

        values = signal_values[int(iidx_start):int(iidx_stop)]

        out_signal = EvenlySignal(values=values,
                                  sampling_freq=self.get_sampling_freq(),
                                  signal_nature=self.get_signal_nature(),
                                  start_time=self.get_time(iidx_start))

        return out_signal

    def to_csv(self, filename, comment=''):
        values = self.get_values()
        times = self.get_times()
        header = self.get_signal_nature() + ' \n' + 'Fsamp: ' + str(
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
    signal_nature : str, default = ''
        Type of signal (e.g. 'ECG', 'EDA')
    start_time: float,
        Instant of signal start
    
    x_values : numpy.array of int
        Instants, or indices when the values are measured.
    x_type : str
        Type of x values given.
        Can be 'indices' or 'instants'
    """

    _MT_X_INDICES = "x_values"

    def __new__(cls, values, sampling_freq=1000, signal_nature="", start_time=None, x_values=None, x_type='instants'):
        assert x_values is not None, "x_values are missing"
        assert x_type in ['indices', 'instants'], "x_type not in ['indices', 'instants']"
        assert len(x_values) == len(values), "Length mismatch (y:%d vs. x:%d)" % (len(values), len(x_values))
        assert len(_np.where(_np.diff(x_values) <= 0)[0]) == 0, 'Given x_values are not strictly monotonic'

        if x_type == 'indices':
            x_values = _np.array(x_values)
            if start_time is None:
                start_time = 0
            else:
                assert start_time < x_values[1] / sampling_freq, "More than one sample at or before start_time"
        else:
            if start_time is None:
                start_time = x_values[0]
                x_values = _np.floor((x_values - start_time) * sampling_freq).astype(int)
            else:
                # TODO Andrea Start time could be also after the first sample so that a signal could be defined also in
                # the start_time instant
                assert start_time < x_values[1], "More than one sample at or before start_time"
                x_values = _np.floor(x_values * sampling_freq).astype(int)

        obj = Signal.__new__(cls, values=values,
                             sampling_freq=sampling_freq,
                             start_time=start_time,
                             signal_nature=signal_nature)
        obj.ph[cls._MT_X_INDICES] = x_values
        return obj

    def get_end_time(self):
        return self.get_time(self.get_indices()[-1]) + 1. / self.get_sampling_freq()  # TODO if good to add 1/f

    def get_times(self):
        return self.ph[self._MT_X_INDICES] / self.get_sampling_freq() + self.get_start_time()

    def get_time(self, idx):
        return idx / self.get_sampling_freq() + self.get_start_time() if idx is not None else None

    def get_indices(self):
        return self.ph[self._MT_X_INDICES]

    def get_time_from_iidx(self, iidx):
        return self.get_indices()[int(iidx)] / self.get_sampling_freq() + self.get_start_time()\
            if iidx < len(self) else None

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
        header = self.get_signal_nature() + ' \n' + 'Fsamp: ' + str(
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

        x_out = _np.arange(data_x[0], data_x[-1] + 1)
        sig_out = tck(x_out)

        # Init new signal
        sig_out = EvenlySignal(values=sig_out,
                               sampling_freq=self.get_sampling_freq(),
                               signal_nature=self.get_signal_nature(),
                               start_time=self.get_start_time())
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
            idx_stop = len(self)
        if idx_start is None:
            idx_start = 0

        iidx_start = int(self.get_iidx_from_idx(idx_start))
        iidx_stop = int(self.get_iidx_from_idx(idx_stop))

        return UnevenlySignal(values=self.get_values()[iidx_start:iidx_stop],
                              x_values=self.get_indices()[iidx_start:iidx_stop],
                              sampling_freq=self.get_sampling_freq(),
                              signal_nature=self.get_signal_nature(),
                              start_time=self.get_time(idx_start),
                              x_type='indices')

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

        return UnevenlySignal(values=self.get_values()[int(iidx_start):int(iidx_stop)],
                              x_values=self.get_indices()[int(iidx_start):int(iidx_stop)],
                              sampling_freq=self.get_sampling_freq(),
                              signal_nature=self.get_signal_nature(),
                              start_time=self.get_time_from_iidx(iidx_start),
                              x_type='indices')

    def __repr__(self):
        return Signal.__repr__(self)[:-1] + " time resolution:" + str(
            1 / self.get_sampling_freq()) + "s>\n" + self.view(
            _np.ndarray).__repr__() + " Times\n:" + self.get_times().__repr__()
