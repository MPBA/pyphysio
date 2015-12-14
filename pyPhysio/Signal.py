# coding=utf-8
from numpy import array, float64

__author__ = 'AleB'


class Signal(object):
    NP_TIME_T = float64

    def __init__(self, signal_nature, start_time):
        assert self.__class__ != Signal.__class__, "Abstract"
        self._signal_type = signal_nature
        self._start_time = start_time
        self._metadata = {}

    @property
    def signal_nature(self):
        return self._signal_type

    @property
    def start_time(self):
        return self._start_time

    @property
    def metadata(self):
        return self._metadata

    def __repr__(self):
        return "<" + self.signal_nature + " signal from:" + self.start_time + ">"


class KFreqSignal(Signal):
    def __init__(self, p_object, sampling_freq, signal_nature, start_time):
        Signal.__init__(self, signal_nature, start_time)
        self._sampling_freq = sampling_freq
        self._data = array(p_object, order="C", ndmin=1)

    @property
    def sampling_freq(self):
        return self._sampling_freq

    @property
    def data(self):
        return self._data

    def __repr__(self):
        return Signal.__repr__(self)[:-1] + " freq:" + self.sampling_freq + "Hz>" + self.data.__repr__()


class IntervalSeries(Signal):
    def __init__(self, intervals, indexes, base_signal):
        Signal.__init__(self, None, None)
        self._intervals = array(intervals, dtype=self.NP_TIME_T, ndmin=1)
        self._indexes = array(indexes, ndmin=1)
        self._base_signal = base_signal

    @property
    def times(self):
        return self._intervals

    @property
    def values(self):
        return self._indexes

    @property
    def signal_nature(self):
        return self._base_signal.signal_nature + "interval"

    @property
    def start_time(self):
        return self._base_signal.start_time


class EventsSignal(Signal):
    def __init__(self, times, values):
        Signal.__init__(self, "events", times[0])
        self._times = array(times, dtype=Signal.NP_TIME_T, ndmin=1)
        self._values = array(values, ndmin=1)

    @property
    def times(self):
        return self._times

    @property
    def values(self):
        return self._values

    def __repr__(self):
        return Signal.__repr__(self) + self.values.__repr__()