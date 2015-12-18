# coding=utf-8
from __future__ import division
from numpy import array, float64, searchsorted

__author__ = 'AleB'


class Signal(object):
    NP_TIME_T = float64

    def __init__(self, signal_nature, start_time, name=None, meta=None):
        assert self.__class__ != Signal.__class__, "Abstract"
        self._signal_type = signal_nature
        self._start_time = start_time
        self._name = name
        self._metadata = meta if meta is not None else {}

    @property
    def signal_nature(self):
        return self._signal_type

    @property
    def start_time(self):
        return self._start_time

    @property
    def duration(self):
        assert self.__class__ != Signal.__class__, "Abstract"
        return None

    @property
    def metadata(self):
        return self._metadata

    def __repr__(self):
        return "<" + self.signal_nature + " signal from:" + self.start_time + ">"

    def getslice(self, f, l):
        assert self.__class__ != Signal.__class__, "Abstract"


class KFreqSignal(Signal):
    def __init__(self, p_object, sampling_freq, signal_nature, start_time):
        Signal.__init__(self, signal_nature, start_time)
        self._sampling_freq = sampling_freq
        self._data = array(p_object, order="C", ndmin=1)

    @property
    def duration(self):
        # Uses future division
        return self.data / self.sampling_freq

    @property
    def sampling_freq(self):
        return self._sampling_freq

    @property
    def data(self):
        return self._data

    def __repr__(self):
        return Signal.__repr__(self)[:-1] + " freq:" + self.sampling_freq + "Hz>" + self.data.__repr__()

    def getslice(self, f, l):
        # find base_signal's indexes
        f = (f - self.start_time) / self.sampling_freq
        l = (l - self.start_time) / self.sampling_freq
        # clip the end
        # [:] has exclusive end
        if l > len(self.data):
            l = len(self.data)
        return KFreqSignal(self.data[f:l], self.sampling_freq, self.signal_nature, f)


class IntervalSeries(Signal):
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
        return IntervalSeries(self.times[f:l], self.indexes[f:l], self.base_signal)


class EventsSignal(Signal):
    def __init__(self, times, values):
        Signal.__init__(self, "events", times[0])
        self._times = array(times, dtype=Signal.NP_TIME_T, ndmin=1)
        self._values = array(values, ndmin=1)

    @property
    def duration(self):
        return self.times[-1]

    @property
    def times(self):
        return self._times

    @property
    def values(self):
        return self._values

    def __repr__(self):
        return Signal.__repr__(self) + self.values.__repr__()

    def getslice(self, f, l):
        # find f & l indexes of indexes
        f = searchsorted(self.times, f)
        l = searchsorted(self.times, l)
        return EventsSignal(self.times[f:l], self.values[f:l])