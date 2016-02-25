# coding=utf-8
from __future__ import division
from numpy import ndarray, float64, searchsorted, arange, asarray

__author__ = 'AleB'


class Signal(ndarray):
    NP_TIME_T = float64

    def __new__(cls, input_array, signal_nature="", start_time=0, meta=None):
        # noinspection PyNoneFunctionAssignment
        obj = asarray(input_array).view(cls)
        obj._pyphysio = {
            "signal_nature": signal_nature,
            "start_time": start_time,
            "metadata": meta if meta is not None else {}
        }
        return obj

    def __array_finalize__(self, obj):
        # __new__ called if obj is None
        if obj is not None:
            self._pyphysio = getattr(obj, '_pyphysio', None)

    def __array_wrap__(self, out_arr, context=None):
        # Just call the parent's
        # noinspection PyArgumentList
        return ndarray.__array_wrap__(self, out_arr, context)

    @property
    def ph(self):
        return self._pyphysio

    @property
    def signal_nature(self):
        return self.ph['signal_nature']

    @property
    def start_time(self):
        return self.ph['start_time']

    @property
    def metadata(self):
        return self.ph["metadata"]

    @property
    def duration(self):
        assert self.__class__ != Signal.__class__, "Abstract"
        return None

    @property
    def end_time(self):
        return self.start_time + self.duration

    def get_times(self):
        assert self.__class__ != Signal.__class__, "Abstract"
        return None

    def __repr__(self):
        return "<signal: " + self.signal_nature + ", start_time: " + str(self.start_time) + ">"

    def getslice(self, f, l):
        assert self.__class__ != Signal.__class__, "Abstract"


class EvenlySignal(Signal):
    def __new__(cls, input_array, sampling_freq, signal_nature="", start_time=0, meta=None):
        obj = Signal(input_array, signal_nature, start_time, meta).view(cls)
        obj.ph["sampling_freq"] = sampling_freq
        return obj

    @property
    def duration(self):
        # Uses future division
        # TODO time_unit: time_unit vs frequency_unit
        return len(self) / self.sampling_freq

    @property
    def sampling_freq(self):
        return self.ph["sampling_freq"]

    def get_times(self):
        # Using future division
        # TODO time_unit: time_unit vs frequency_unit
        tmp_step = 1. / self.sampling_freq
        return arange(self.start_time, self.end_time, tmp_step)

    def __repr__(self):
        return Signal.__repr__(self)[:-1] + " freq:" + str(self.sampling_freq) + "Hz>\n" + self.get_values().__repr__()

    # Works with timestamps
    def getslice(self, f, l):
        # Using future division
        # TODO time_unit: time_unit vs frequency_unit
        # find base_signal's indexes
        f = (f - self.start_time) / self.sampling_freq
        l = (l - self.start_time) / self.sampling_freq
        # clip the end
        # [:] has exclusive end
        if l > len(self.get_values()):
            l = len(self.get_values())
        return EvenlySignal(self.get_values()[f:l], self.sampling_freq, self.signal_nature, f)


class UnevenlySignal(Signal):
    def __new__(cls, input_array, times_array, signal_nature="", start_time=0, meta=None, check=True):
        # TODO check: useful O(n) monotonicity check?
        assert not check or len(input_array) == len(times_array),\
            "Length mismatch (%d vs. %d)" % (len(input_array), len(times_array))
        assert not all(times_array[i] <= times_array[i+1] for i in xrange(len(times_array)-1)),\
            "Time is not monotonic"
        obj = Signal(input_array, signal_nature, start_time, meta).view(cls)
        obj.ph["times"] = times_array
        return obj

    def get_times(self):
        return self.ph["times"]

    def __repr__(self):
        return Signal.__repr__(self)\
            + "\ntimes-" + self.get_times().__repr__() + "\nvalues-" + self.get_values().__repr__()

    # Works with timestamps
    def getslice(self, f, l):
        # find f & l indexes of indexes
        f = searchsorted(self._times, f)
        l = searchsorted(self._times, l)
        return UnevenlySignal(self.get_values[f:l], self.times[f:l], self.signal_nature, check=False)


class EventsSignal(UnevenlySignal):
    def __new__(cls, events, times, meta=None, checks=True):
        return UnevenlySignal(events, times, "events", meta, checks)

    # Works with timestamps
    def getslice(self, f, l):
        # find f & l indexes of indexes
        f = searchsorted(self.times, f)
        l = searchsorted(self.times, l)
        return EventsSignal(self.times[f:l], self.get_values[f:l], checks=False)


class UnevenlyPointersSignal(Signal):
    def __init__(self, intervals, indexes, base_signal):
        Signal.__init__(self, None, None)
        self._intervals = array(intervals, dtype=self.NP_TIME_T, ndmin=1)
        self._indexes = array(indexes, ndmin=1)
        self._base_signal = base_signal

    @property
    def duration(self):
        return self.base_signal.duration

    @property
    def times(self):
        return self._intervals

    @property
    def indexes(self):
        return self._indexes

    @property
    def base_signal(self):
        return self._base_signal

    @property
    def signal_nature(self):
        return self.base_signal.signal_nature + "interval"

    @property
    def start_time(self):
        return self.base_signal.start_time

    def getslice(self, f, l):
        # find base_signal's indexes
        f = (f - self.start_time) / self.base_signal.sampling_freq
        l = (l - self.start_time) / self.base_signal.sampling_freq
        # clip the end
        # [:] has exclusive end
        if l > len(self.base_signal.data):
            l = len(self.base_signal.data)
        # find f & l indexes of indexes
        f = searchsorted(self.indexes, f)
        l = searchsorted(self.indexes, l)
        return UnevenlySignal(self.times[f:l], self.indexes[f:l], self.base_signal)