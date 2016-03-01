# coding=utf-8
from ..PhUI import PhUI
from ..BaseSegmentation import SegmentsGenerator, Segment
from ..Signal import EvenlySignal as _EvenlySignal

__author__ = 'AleB'


class LengthSegments(SegmentsGenerator):
    """
    Constant length (samples number) segments
    __init__(self, step, width=0, start=0)
    """

    def __init__(self, params=None, **kwargs):
        super(LengthSegments, self).__init__(params, kwargs)
        assert "step" in self._params, "Need the parameter 'step' for the segmentation."
        self._step = None
        self._width = None
        self._i = None

    def init_segmentation(self):
        if self._signal is None:
            raise ValueError("Can't preview the segments without a signal here. Use the syntax "
                             + LengthSegments.__name__ + "(p[params])(signal)")
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
        super(TimeSegments, self).__init__(params, kwargs)
        assert "step" in self._params, "Need the parameter 'step' for the segmentation."
        self._step = None
        self._width = None
        self._i = None
        self._c_times = None

    def init_segmentation(self):
        if self._signal is None:
            raise ValueError("Can't preview the segments without a signal here. Use the syntax "
                             + LengthSegments.__name__ + "(p[params])(signal)")
        self._step = self._params["step"]
        self._width =\
            self._params["step"] if "width" not in self._params or self._params["width"] == 0 else self._params["width"]
        self._i = self._params["start"] if "start" in self._params else 0
        self._signal = self._signal
        self._c_times = self._signal.get_times()

    def next_segment(self):
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


class ExistingSegments(SegmentsGenerator):
    """
    Wraps a list of windows from an existing collection.
    """

    def __init__(self, win_list):
        """
        Initializes the win generator
        @param win_list: List of Windows to consider
        """
        super(ExistingSegments, self).__init__()

        self._wins = win_list
        self._ind = 0

    def next_segment(self):
        if self._ind >= len(self._wins):
            self._ind = 0
            raise StopIteration
        else:
            self._ind += 1
            PhUI.a(isinstance(self._wins[self._ind - 1], Segment), "%d is not a Segment" % self._wins[self._ind - 1])
            return self._wins[self._ind - 1]

    def init_segmentation(self):
        pass


class LabeledSegments(SegmentsGenerator):
    """
    Generates a list of windows from a labels list.
    """

    def __init__(self, labels, include_baseline_name=None):
        """
        Initializes the win generator
        @param labels: Labels time series
        """
        super(LabeledSegments, self).__init__()
        self._i = 0
        self._ibn = include_baseline_name
        self._labels = labels

    def next_segment(self):
        # TODO limit also using the signal length
        if self._i < len(self._labels) - 1:
            w = Segment(self._labels.index[self._i], self._labels.index[self._i + 1], self._labels.values[self._i],
                        self._signal)
        elif self._i < len(self._labels):
            w = Segment(self._labels.index[self._i], None, self._labels.values[self._i],
                        self._signal)
        else:
            raise StopIteration()
        self._i += 1
        return w

    def init_segmentation(self):
        pass
