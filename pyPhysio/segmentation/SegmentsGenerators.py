# coding=utf-8
from ..Utility import PhUI as _PhUI
from ..BaseSegmentation import SegmentsGenerator, Segment

__author__ = 'AleB'


class LengthSegments(SegmentsGenerator):
    """
    Constant length (samples number) segments
    __init__(self, step, width=0, start=0)
    """

    def __init__(self, params=None, **kwargs):
        super(LengthSegments, self).__init__(params, **kwargs)
        assert "step" in self._params, "Need the parameter 'step' for the segmentation."
        self._step = None
        self._width = None
        self._i = None

    def init_segmentation(self):
        if self._signal is None:
            raise ValueError("Can't preview the segments without a signal here. Use the syntax " +
                             LengthSegments.__name__ + "(**params)(signal)")
        self._step = self._params["step"]
        self._width =\
            self._params["step"] if "width" not in self._params or self._params["width"] == 0 else self._params["width"]
        self._i = self._params["start"] if "start" in self._params else 0
        self._signal = self._signal

    def next_segment(self):
        o = self._i
        self._i += self._step
        s = Segment(o, o + self._width, '', self._signal)
        if s.is_empty():
            raise StopIteration()
        return s


class TimeSegments(SegmentsGenerator):
    """
    Constant length (time) segments
    __init__(self, step, width=0, start=0)
    """
    def __init__(self, params=None, **kwargs):
        super(TimeSegments, self).__init__(params, **kwargs)
        assert "step" in self._params, "Need the parameter 'step' for the segmentation."
        self._step = None
        self._width = None
        self._i = None
        self._c_times = None

    def init_segmentation(self):
        self._step = self._params["step"]
        self._c_times = self._signal.get_times()
        self._width =\
            self._params["step"] if "width" not in self._params or self._params["width"] == 0 else self._params["width"]
        self._i = 0
        # initial seek
        if "start" in self._params:
            start = self._params["start"]
            while self._i < len(self._signal) and self._signal.get_indices(self._i) < start:
                self._i += 1

    def next_segment(self):
        if self._signal is None:
            _PhUI.w("Can't preview the segments without a signal here. Use the syntax " +
                    TimeSegments.__name__ + "(p[params])(signal)")
            raise StopIteration()
        b = e = self._i
        l = len(self._signal)
        while self._i < l and self._c_times[self._i] <= self._c_times[b] + self._step:
            self._i += 1
        while e < l and self._c_times[e] <= self._c_times[b] + self._width:
            e += 1
        s = Segment(b, e, '', self._signal)
        if s.is_empty():
            raise StopIteration()
        return s


class FromStartStopSegments(SegmentsGenerator):
    """
    Constant length (time) segments
    __init__(self, step, width=0, start=0)
    """
    def __init__(self, params=None, **kwargs):
        super(FromStartStopSegments, self).__init__(params, **kwargs)
        assert "starts" in self._params, "Need the parameter 'start' (array of times) for the segmentation."
        assert "stops" in self._params, "Need the parameter 'stop' (array of times) for the segmentation."
        self._b = None
        self._e = None
        self._i = None

    def init_segmentation(self):
        self._b = 0
        self._e = 0
        self._i = 0

    def next_segment(self):
        if self._signal is None:
            _PhUI.w("Can't preview the segments without a signal here. Use the syntax "
                 + TimeSegments.__name__ + "(p[params])(signal)")
            raise StopIteration()
        else:
            if self._i < len(self._params['starts']):
                l = len(self._signal)
                start = self._params['starts'][self._i]
                while self._b < l and self._signal.get_indices(self._b) < start:
                    self._b += 1
                stop = self._params['stops'][self._i]
                while self._e < l and self._signal.get_indices(self._e) < stop:
                    self._e += 1

                self._i += 1

                s = Segment(self._b, self._e, '', self._signal)

                if s.is_empty():
                    raise StopIteration()
                else:
                    return s
            else:
                raise StopIteration()


class ExistingSegments(SegmentsGenerator):
    """
    Wraps a list of windows from an existing collection.
    """

    def __init__(self, params=None, **kwargs):
        super(ExistingSegments, self).__init__(params, **kwargs)
        assert "segments" in self._params, "Need the parameter 'segments' (array of Segment) for the segmentation."
        self._wins = None
        self._ind = None

    def init_segmentation(self):
        self._wins = iter(self._params["segments"])
        self._ind = 0

    def next_segment(self):
        w = self._wins.next()
        if self._signal is not None:
            assert isinstance(w, Segment), "%s is not a Segment" % str(w)
            w = Segment(w.get_begin(), w.get_end(), w.get_label(), self._signal)
            if w.is_empty():
                raise StopIteration()
        return w


class FromEventsSegments(SegmentsGenerator):
    """
    Generates a list of windows from a labels list.
    """

    def __init__(self, params=None, **kwargs):
        super(FromEventsSegments, self).__init__(params, **kwargs)
        assert "events" in self._params, "Need the parameter 'events' for this segmentation."
        self._i = None
        self._t = None
        self._s = None
        self._ibn = None
        self._events = None
        self._c_times = None

    def init_segmentation(self):
        self._ibn = self._params["include_baseline_name"] if "include_baseline_name" in self._params else None
        self._events = self._params["events"]
        self._s = 0
        self._i = 0
        self._t = self._events.get_indices(0)

        # TESTME: May be not so efficient but it is better than searchsorted (small k < n often smaller than log2(n))
        while self._i < len(self._signal) and self._signal.get_indices(self._i) < self._t:
            self._i += 1

    def next_segment(self):
        if self._signal is None:
            _PhUI.w("Can't preview the segments without a signal here. Use the syntax "
                    + LengthSegments.__name__ + "(**params)(signal)")
            raise StopIteration()
        else:
            l = len(self._signal)
            if self._i < l:
                if self._s < len(self._events) - 1:
                    o = self._i
                    self._t = self._events.get_indices(self._s + 1)
                    while self._i < l and self._signal.get_indices(self._i) < self._t:
                        self._i += 1
                    w = Segment(o, self._i, self._events[self._s], self._signal)
                elif self._s < len(self._events):
                    w = Segment(self._i, l, self._events[self._s], self._signal)
                else:
                    raise StopIteration()
            else:
                raise StopIteration()
        self._s += 1
        return w
