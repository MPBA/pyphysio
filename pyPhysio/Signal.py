# coding=utf-8
from __future__ import division
import numpy as _np
from abc import abstractmethod as _abstract, ABCMeta as _ABCMeta

__author__ = 'AleB'

# Everything in SECONDS (s) !!!


class Signal(_np.ndarray):
    __metaclass__ = _ABCMeta

    _MT_NATURE = "signal_nature"
    _MT_START_TIME = "start_time"
    _MT_META_DICT = "metadata"
    _MT_SAMPLING_FREQ = "sampling_freq"
    _MT_INFO_ATTR = "_pyphysio"

    @_abstract
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
        return None

    @_abstract
    def get_x_values(self, just_one=None):
        return None

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


class EvenlySignal(Signal):
    def __new__(cls, y_values, sampling_freq, signal_nature="", start_time=0, meta=None):
        obj = Signal(y_values, sampling_freq, signal_nature, start_time, meta).view(cls)
        return obj

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
        return Signal.__repr__(self)[:-1] + " freq:" + str(self.get_sampling_freq()) + "Hz>\n" + self.view(
            _np.ndarray).__repr__()


class XYSignal(Signal):
    __metaclass__ = _ABCMeta

    _MT_X_VALUES = "x_values"

    @_abstract
    def __new__(cls, y_values, x_values, sampling_freq, signal_nature, start_time, meta, check):
        assert not check or len(y_values) == len(x_values), \
            "Length mismatch (y:%d vs. x:%d)" % (len(y_values), len(x_values))
        obj = Signal(y_values, sampling_freq, signal_nature, start_time, meta).view(cls)
        obj.ph[cls._MT_X_VALUES] = x_values
        return obj

    def get_x_values(self, just_one=None):
        if just_one is None:
            return self.ph[self._MT_X_VALUES]
        else:
            return self.ph[self._MT_X_VALUES][just_one]

    @_abstract
    def get_duration(self):
        return None

    @_abstract
    def getslice(self, f, l):
        pass

    def __repr__(self):
        return Signal.__repr__(self) + "\ntimes-" + self.get_x_values().__repr__() + "\nvalues-" + self.view(
            _np.ndarray).__repr__()


class UnevenlySignal(XYSignal):
    _MT_DURATION = "duration"

    def __new__(cls, y_values, time_values, duration, sampling_freq, signal_nature="", start_time=0, meta=None,
                check=True):
        obj = XYSignal(y_values, time_values, sampling_freq, signal_nature, start_time, meta, check).view(cls)
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


class UnevenlyTimeSignal(XYSignal):
    def __new__(cls, y_values, time_values, signal_nature="", start_time=0, meta=None, check=True):
        return XYSignal(y_values, time_values, signal_nature, start_time, meta, check).view(cls)

    def get_duration(self):
        return self.get_start_time() + self.get_x_values(len(self))

    # Works with timestamps
    def getslice(self, f, l):
        # find f & l indexes of indexes
        f = _np.searchsorted(self.get_x_values(), f)
        l = _np.searchsorted(self.get_x_values, l)
        return UnevenlySignal(self[f:l], self.get_x_values()[f:l], self.get_signal_nature(), check=False)


class EventsSignal(UnevenlyTimeSignal):
    def __new__(cls, events, times, start_time=0, meta=None, check=True):
        return UnevenlySignal(events, times, 0, 0, "events", start_time, meta, check)

    # Works with timestamps
    def getslice(self, f, l):
        # find f & l indexes of indexes
        f = _np.searchsorted(self.get_x_values(), f)
        l = _np.searchsorted(self.get_x_values(), l)
        return EventsSignal(self.get_x_values()[f:l], self.view(_np.ndarray)[f:l], check=False)
