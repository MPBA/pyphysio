# coding=utf-8
from __future__ import division
import numpy as _np
from scipy import interpolate as _interp
from Utility import abstractmethod as _abstract, PhUI as _PhUI
from matplotlib.pyplot import plot as _plot

__author__ = 'AleB'

# Everything in SECONDS (s) !!!


class Signal(_np.ndarray):
    _MT_NATURE = "signal_nature"
    _MT_START_TIME = "start_time"
    _MT_START_INDEX = "start_index"
    _MT_META_DICT = "metadata"
    _MT_SAMPLING_FREQ = "sampling_freq"
    _MT_INFO_ATTR = "_pyphysio"

    def __new__(cls, values, sampling_freq, signal_nature="", start_time=0, meta=None, start_index=0):
        # noinspection PyNoneFunctionAssignment
        obj = _np.asarray(values).view(cls)
        obj._pyphysio = {
            cls._MT_NATURE: signal_nature,
            cls._MT_START_TIME: start_time,
            cls._MT_START_INDEX: start_index,
            cls._MT_SAMPLING_FREQ: sampling_freq,
            cls._MT_META_DICT: meta if meta is not None else {}
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
    def get_indices(self, just_one=None):
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
        return self.get_start_time() + self.get_duration()

    def get_metadata(self):
        return self.ph[self._MT_META_DICT]

    def plot(self, style=""):
        _plot(self.get_indices(), self.get_values(), style)

    def __repr__(self):
        return "<signal: " + self.get_signal_nature() + ", start_time: " + str(self.get_start_time()) + ">"


class EvenlySignal(Signal):
    def get_duration(self):
        # Uses future division
        return len(self) / self.get_sampling_freq()

    def get_indices(self, just_one=None):
        # Using future division
        if just_one is None:
            return _np.arange(len(self)) + self.ph[Signal._MT_START_INDEX]
        else:
            if just_one >= 0:
                return just_one + self.ph[Signal._MT_START_INDEX]
            else:
                return just_one + self.ph[Signal._MT_START_INDEX] + len(self)

    def get_times(self, just_one=None):
        return self.get_indices(just_one) / self.get_sampling_freq()

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
            signal_out = self.get_values()[::ratio]
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

        return EvenlySignal(signal_out, fout, self.get_signal_nature(), self.get_start_time(), self.get_metadata())

    def __getslice__(self, i, j):
        o = Signal.__getslice__(self, i, j)
        if isinstance(o, Signal):
            o.ph[Signal._MT_START_INDEX] += i
        return o


class UnevenlySignal(Signal):
    _MT_X_VALUES = "x_values"
    _MT_ORIGINAL_LENGTH = "original_length"

    def __new__(cls, values, indices, orig_sampling_freq, orig_length, signal_nature="", start_time=0, meta=None, check=True):
        assert not check or len(values) == len(indices), \
            "Length mismatch (y:%d vs. x:%d)" % (len(values), len(indices))
        indices = _np.array(indices)
        # assert not check or x_values.all(x_values.argsort()), \
        #     "x_values array not monotonic."
        obj = Signal.__new__(cls, values, orig_sampling_freq, signal_nature, start_time, meta)
        obj.ph[cls._MT_X_VALUES] = indices
        obj.ph[cls._MT_ORIGINAL_LENGTH] = orig_length
        return obj

    def get_indices(self, just_one=None):
        if just_one is None:
            return self.ph[self._MT_X_VALUES]
        else:
            return self.ph[self._MT_X_VALUES][just_one]

    def get_times(self, just_one=None):
        self.get_indices(just_one) / self.get_sampling_freq()

    def __repr__(self):
        return Signal.__repr__(self) + "\ny-values\n" + self.view(_np.ndarray).__repr__() + \
            "\nx-indices\n" + self.get_indices().__repr__()

    def __getslice__(self, i, j):
        o = Signal.__getslice__(self, i, j)
        if isinstance(o, UnevenlySignal):
            o.ph[UnevenlySignal._MT_X_VALUES] = o.ph[UnevenlySignal._MT_X_VALUES].__getslice__(i, j)
        return o

    def get_duration(self):
        return self.ph[self._MT_ORIGINAL_LENGTH] / self.get_sampling_freq()

    def get_original_length(self):
        return self.ph[self._MT_ORIGINAL_LENGTH]

    def to_evenly(self, kind='cubic', length=None):
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

        length = self.get_original_length() if length is None else length

        data_x = self.get_indices()  # From a constant freq range
        # check that the computed length is bigger than the data_x one
        if length < len(data_x):
            _PhUI.w("Signal: the original signal is shorter than the current (that should be a subset, so that original_length <= len(get_indices()) but this is false).")
        data_y = self.get_values()
        #print(len(data_x))
        #print(len(data_y))

        # Add padding
        if self.get_indices(0) != 0:
            data_x = _np.r_[0, data_x]
            data_y = _np.r_[data_y[0], data_y]
        if self.get_indices(-1) != length - 1:
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

    # # ex sparse
    # def zto_evenly(self, kind='linear', new_fsamp=None, length=None):
    #     """
    #     Interpolate the UnevenlySignal to obtain an evenly spaced signal
    #     Parameters
    #     ----------
    #     kind : str
    #         Method for interpolation: 'linear', 'nearest', 'zero', 'slinear', 'quadratic, 'cubic'
    #
    #     new_fsamp : number
    #         Frequency of the resulting signal. If not specified the original signal's will be used.
    #
    #     length : number
    #         Length in samples of the resulting signal. If not specified the last sample will be one after the last input point.
    #
    #     Returns
    #     -------
    #     interpolated_signal: ndarray
    #         The interpolated signal
    #     """
    #
    #     if new_fsamp is None:
    #         new_fsamp = self.get_sampling_freq()
    #     if length is None:
    #         length = self.get_duration() * new_fsamp
    #     data_x = self.get_indices()
    #     data_y = self.get_values()
    #
    #     # Add padding
    #     if self.get_indices(0) != 0:
    #         data_x = _np.r_[0, data_x]
    #         data_y = _np.r_[data_y[0], data_y]
    #     if self.get_indices(-1) != length - 1:
    #         data_x = _np.r_[data_x, length - 1]
    #         data_y = _np.r_[data_y, data_y[-1]]
    #
    #     # Cubic if needed
    #     if kind == 'cubic':
    #         tck = _interp.InterpolatedUnivariateSpline(data_x, data_y)
    #     else:
    #         tck = _interp.interp1d(data_x, data_y, kind=kind)
    #     sig_out = tck(_np.arange(length))
    #
    #     # Init new signal
    #     sig_out = EvenlySignal(sig_out, new_fsamp, self.get_signal_nature(), self.get_start_time(),
    #                            self.get_metadata())
    #     return sig_out


class EventsSignal(UnevenlySignal):
    def __new__(cls, values, times, orig_sampling_freq=1, orig_length=None, signal_nature="", start_time=0,
                meta=None, check=True):
        return UnevenlySignal.__new__(cls, values, times * orig_sampling_freq, orig_sampling_freq, orig_length, signal_nature, start_time,
                                      meta, check)


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
