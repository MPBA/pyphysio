# coding=utf-8
from __future__ import division
import numpy as _np

__author__ = 'AleB'

# Everything in SECONDS (s) !!!


class Signal(_np.ndarray):
    _NP_TIME_T = _np.float64
    _MT_NATURE = "signal_nature"
    _MT_START_TIME = "start_time"
    _MT_META_DICT = "metadata"
    _MT_INFO_ATTR = "_pyphysio"

    def __new__(cls, input_array, signal_nature="", start_time=0, meta=None):
        # noinspection PyNoneFunctionAssignment
        obj = _np.asarray(input_array).view(cls)
        obj._pyphysio = {
            cls._MT_NATURE: signal_nature,
            cls._MT_START_TIME: start_time,
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

    @property
    def signal_nature(self):
        return self.ph[self._MT_NATURE]

    @property
    def start_time(self):
        return self.ph[self._MT_START_TIME]

    @property
    def metadata(self):
        return self.ph[self._MT_META_DICT]

    @property
    def duration(self):
        assert self.__class__ != Signal.__class__, "Abstract"
        return None

    @property
    def end_time(self):
        return self.start_time + self.duration

    def get_times(self, just_one=None):
        assert self.__class__ != Signal.__class__, "Abstract"
        return None

    def __repr__(self):
        return "<signal: " + self.signal_nature + ", start_time: " + str(self.start_time) + ">"

    def getslice(self, f, l):
        assert self.__class__ != Signal.__class__, "Abstract"


class EvenlySignal(Signal):
    _MT_SAMPLING_FREQ = "sampling_freq"

    def __new__(cls, input_array, sampling_freq, signal_nature="", start_time=0, meta=None):
        obj = Signal(input_array, signal_nature, start_time, meta).view(cls)
        obj.ph[cls._MT_SAMPLING_FREQ] = sampling_freq
        return obj

    @property
    def duration(self):
        # Uses future division
        return len(self) / self.sampling_freq

    @property
    def sampling_freq(self):
        return self.ph[self._MT_SAMPLING_FREQ]

    def get_times(self, just_one=None):
        # Using future division
        tmp_step = 1. / self.sampling_freq
        if just_one is None:
            return _np.arange(self.start_time, self.end_time, tmp_step)
        else:
            return self.start_time + tmp_step * just_one

    def __repr__(self):
        return Signal.__repr__(self)[:-1] + " freq:" + str(self.sampling_freq) + "Hz>\n"\
            + self.view(_np.ndarray).__repr__()

    # Works with timestamps
    def getslice(self, f, l):
        # Using future division
        # find base_signal's indexes
        f = (f - self.start_time) / self.sampling_freq
        l = (l - self.start_time) / self.sampling_freq
        # clip the end
        # [:] has exclusive end
        if l > len(self):
            l = len(self)
        return EvenlySignal(self[f:l], self.sampling_freq, self.signal_nature, f)


class UnevenlySignal(Signal):
    _MT_TIMES = "times"

    def __new__(cls, input_array, times_array, signal_nature="", start_time=0, meta=None, check=True):
        # TODO check: useful O(n) monotonicity check?
        assert not check or len(input_array) == len(times_array),\
            "Length mismatch (%d vs. %d)" % (len(input_array), len(times_array))
        assert all(i > 0 for i in _np.diff(times_array)), "Time is not monotonic"
        obj = Signal(input_array, signal_nature, start_time, meta).view(cls)
        obj.ph[cls._MT_TIMES] = times_array
        return obj

    def get_times(self, just_one=None):
        if just_one is None:
            return self.ph[self._MT_TIMES]
        else:
            return self.ph[self._MT_TIMES][just_one]

    def __repr__(self):
        return Signal.__repr__(self)\
            + "\ntimes-" + self.get_times().__repr__() + "\nvalues-" + self.view(_np.ndarray).__repr__()

    # Works with timestamps
    def getslice(self, f, l):
        # find f & l indexes of indexes
        f = _np.searchsorted(self.get_times(), f)
        l = _np.searchsorted(self.get_times, l)
        return UnevenlySignal(self[f:l], self.get_times()[f:l], self.signal_nature, check=False)


class EventsSignal(UnevenlySignal):
    def __new__(cls, events, times, meta=None, checks=True):
        return UnevenlySignal(events, times, "events", meta, checks)

    # Works with timestamps
    def getslice(self, f, l):
        # find f & l indexes of indexes
        f = _np.searchsorted(self.get_times(), f)
        l = _np.searchsorted(self.get_times(), l)
        return EventsSignal(self.get_times()[f:l], self.view(_np.ndarray)[f:l], checks=False)
