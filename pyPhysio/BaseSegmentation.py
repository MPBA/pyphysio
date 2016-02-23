# coding=utf-8
__author__ = 'AleB'
__all__ = ['SegmentationError', 'Segment', 'SegmentsGenerator']
from copy import copy as cpy


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
        return self._signal.times[self.begin]  # TODO: add prop to base signal to avoid arange computation?

    @property
    def end_time(self):
        return self._signal.times[self.end] if self.end is not None else None

    @property
    def duration(self):
        return (self.end_time - self.start_time) if self.end is not None else None

    @property
    def label(self):
        return self._label

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
        return '%s:%s:%s' % (str(self.begin), str(self.end), self._label)


class SegmentsGenerator(object):
    """
    Base and abstract class for the windows computation.
    """

    def __iter__(self):
        return SegmentationIterator(self)

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


class SegmentationIterator(object):
    """
    A generic iterator that is called from each WindowGenerator from the __iter__ method.
    """

    def __init__(self, win):
        assert isinstance(win, SegmentsGenerator)
        self._win = cpy(win)
        self._win.init_segmentation()

    def next(self):
        return self._win.next_segment()


class SegmentationError(Exception):
    """
    Generic "Windowing" error.
    """
    pass
