# coding=utf-8
__author__ = 'AleB'
__all__ = ['TimeSegments', 'LabeledSegments', 'ExistingSegments']

from ..BaseSegmentation import SegmentsGenerator, Segment
from ..PhUI import PhUI


class TimeSegments(SegmentsGenerator):
    """
    Linear-timed segments
    """
    def __init__(self, step, width=0, start=0):
        super(TimeSegments, self).__init__()
        self._step = step
        self._width = step if width == 0 else width
        self._i = start

    def next_segment(self):
        o = self._i
        self._i += self._step
        return Segment(o, o + self._width, '', None)  # TODO: where does it come from?

    def init_segmentation(self):
        pass


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
        if self._i < len(self._labels) - 1:
            w = Segment(self._labels.index[self._i], self._labels.index[self._i + 1], self._labels.values[self._i],
                        None)  # TODO: where does it come from?
        elif self._i < len(self._labels):
            w = Segment(self._labels.index[self._i], None, self._labels.values[self._i],
                        None)  # TODO: where does it come from?
        else:
            raise StopIteration()
        self._i += 1
        return w

    def init_segmentation(self):
        pass
