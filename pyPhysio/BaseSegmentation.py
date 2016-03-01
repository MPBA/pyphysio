# coding=utf-8
from copy import copy as _cpy
import numpy as _np
from BaseAlgorithm import Algorithm as _Algorithm
from Signal import EvenlySignal as _EvenlySignal
__author__ = 'AleB'


class Segment(object):
    """
    Base Segment, a begin-end pair with a reference to the base signal and a name.
    """

    def __init__(self, begin, end, label, signal):
        """
        Creates a base Window
        @param begin: Begin sample index
        @param end: End sample index
        """
        self._begin = begin
        self._end = end
        self._label = label
        self._signal = signal

    @property
    def begin(self):
        return self._begin

    @property
    def end(self):
        return self._end

    @property
    def start_time(self):
        return self._signal.start_time + self._signal.get_times(self._begin)  # TODO this does not work

    @property
    def end_time(self):
        return self._signal.get_times(self.end) if self.end is not None else None

    @property
    def duration(self):
        return (self.end_time - self.start_time) if self.end is not None else None

    @property
    def label(self):
        return self._label

    def is_empty(self):
        return self._signal is None or self._begin >= len(self._signal)

    from datetime import datetime as dt, MAXYEAR
    _mdt = dt(MAXYEAR, 12, 31, 23, 59, 59, 999999)

    def __call__(self, data):
        if self._end is None:
            return data.between_time(self._begin, Segment._mdt)
        else:
            return data.between_time(self._begin, self._end)

    def islice(self, data, include_partial=False):
        if (include_partial or self._end <= data.index[-1]) and self._begin < data.index[-1]:
            return self(data)
        else:
            raise StopIteration()

    def __repr__(self):
        return '%s:%s' % (str(self.begin), str(self.end)) + (":%s" % self._label) if self._label is not None else ""


class SegmentsGenerator(_Algorithm):
    """
    Base and abstract class for the windows computation.
    """

    def __init__(self, params=None, p_kwargs=None, **kwargs):
        if p_kwargs is not None:
            kwargs.update(p_kwargs)
        super(SegmentsGenerator, self).__init__(params, kwargs)
        self._signal = None

    # Algorithm Override
    def __call__(self, data):
        return self.get(data, self._params, use_cache=False)

    def __iter__(self):
        return SegmentationIterator(self)

    @classmethod
    def algorithm(cls, data, params):
        o = cls(params)
        o._signal = data
        return o

    @classmethod
    def is_nature_supported(cls, data):
        return isinstance(data, _EvenlySignal)

    @classmethod
    def get_used_params(cls):
        return []

    def next_segment(self):
        """
        Executes a segmentation step.
        @raise StopIteration: End of the iteration
        """
        raise StopIteration()

    def init_segmentation(self):
        """
        Executes a segmentation step.
        @raise StopIteration: End of the iteration
        """
        raise NotImplementedError()

    def __repr__(self):
        if self._signal is not None:
            return super(SegmentsGenerator, self).__repr__() + " over\n" + str(self._signal)
        else:
            return super(SegmentsGenerator, self).__repr__()


class SegmentationIterator(object):
    """
    A generic iterator that is called from each WindowGenerator from the __iter__ method.
    """

    def __init__(self, win):
        assert isinstance(win, SegmentsGenerator)
        self._win = _cpy(win)
        self._win.init_segmentation()

    def next(self):
        return self._win.next_segment()


class SegmentationError(Exception):
    """
    Generic Segmentation error.
    """
    pass
