# coding=utf-8
from __future__ import division
from numpy import array, float64, searchsorted, arange

__author__ = 'AleB'


class Signal(object):
    NP_TIME_T = float64

    def __init__(self, signal_nature, start_time, meta=None):
        assert self.__class__ != Signal.__class__, "Abstract"
        self._signal_type = signal_nature
        self._start_time = start_time
        self._metadata = meta if meta is not None else {}
        self._data = None

    @property
    def signal_nature(self):
        return self._signal_type

    @property
    def start_time(self):
        return self._start_time

    @property
    def end_time(self):
        return self._start_time + self.duration

    @property
    def duration(self):
        assert self.__class__ != Signal.__class__, "Abstract"
        return None

    @property
    def metadata(self):
        return self._metadata

    @property
    def values(self):
        return self.get_values()

    @values.setter
    def values(self, values):
        self.set_values(values)

    def get_values(self):
        assert self.__class__ != Signal.__class__, "Abstract"
        return self._data

    def set_values(self, values):
        assert self.__class__ != Signal.__class__, "Abstract"
        self._data = array(values, order="C", ndmin=1)

    @property
    def times(self):
        assert self.__class__ != Signal.__class__, "Abstract"
        return None

    def __repr__(self):
        return "<signal: " + self.signal_nature + ", start_time: " + str(self.start_time) + ">"

    def getslice(self, f, l):
        assert self.__class__ != Signal.__class__, "Abstract"


class EvenlySignal(Signal):
    def __init__(self, values, sampling_freq, signal_nature, start_time, meta=None):
        Signal.__init__(self, signal_nature, start_time, meta)
        self._sampling_freq = sampling_freq
        self.set_values(values)

    @property
    def duration(self):
        # Uses future division
        # TODO time_unit: time_unit vs frequency_unit
        return len(self.get_values()) / self.sampling_freq

    @property
    def sampling_freq(self):
        return self._sampling_freq

    @property
    def times(self):
        # Using future division
        # TODO time_unit: time_unit vs frequency_unit
        tmp_step = 1. / self.sampling_freq
        return arange(self.start_time, self.end_time, tmp_step)

    def __repr__(self):
        return Signal.__repr__(self)[:-1] + " freq:" + str(self.sampling_freq) + "Hz>" + self.get_values().__repr__()

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


class UnevenlySignal(Signal):
    def __init__(self, times, values, signal_nature, meta=None, checks=True):
        Signal.__init__(self, signal_nature, times[0], meta)
        # TODO check: useful O(n) monotonicity check?
        assert not checks or all(times[i] <= times[i+1] for i in xrange(len(times)-1))
        self._times = array(times, dtype=self.NP_TIME_T, ndmin=1)
        self.set_values(values)

    @property
    def times(self):
        return self._times

    def __repr__(self):
        return Signal.__repr__(self) + self.get_values.__repr__()

    # Works with timestamps
    def getslice(self, f, l):
        # find f & l indexes of indexes
        f = searchsorted(self._times, f)
        l = searchsorted(self._times, l)
        return UnevenlySignal(self.times[f:l], self.get_values[f:l], self.signal_nature, checks=False)


class EventsSignal(UnevenlySignal):
    def __init__(self, times, values, meta=None, checks=True):
        UnevenlySignal.__init__(self, times, values, "events", meta, checks)

    # Works with timestamps
    def getslice(self, f, l):
        # find f & l indexes of indexes
        f = searchsorted(self.times, f)
        l = searchsorted(self.times, l)
        return EventsSignal(self.times[f:l], self.get_values[f:l], checks=False)