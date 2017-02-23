# coding=utf-8
from ..Utility import PhUI as _PhUI
from ..BaseSegmentation import SegmentsGenerator, Segment
from ..Signal import UnevenlySignal as _UnevenlySignal
from pyphysio.Utility import abstractmethod as _abstract

__author__ = 'AleB'


# WAS: class LengthSegments(SegmentsGenerator):

class _SegmentsWithLabelSignal(SegmentsGenerator):

    def __init__(self, drop_shorter=True, **kwargs):
        super(_SegmentsWithLabelSignal, self).__init__(drop_shorter=drop_shorter, **kwargs)
        self._labsig = None

    @_abstract
    def init_segmentation(self):
        pass

    def next_segment(self):
        if self._signal is None:
            _PhUI.w("Can't preview the segments without a signal here. Use the syntax " +
                    FixedSegments.__name__ + "(**params)(signal)")
            raise StopIteration()

        b, e, label = self.next_segment_mix_labels()
        s = Segment(b, e, label, self._signal)
        return s

    @_abstract
    def next_times(self):
        pass

    def next_segment_mix_labels(self):
        label = b = e = None
        while True:
            b, e = self.next_times()

            # signal segment bounds
            first_idx = self._signal.get_idx(b) if b is not None else b
            last_idx = self._signal.get_idx(e) if e is not None else e

            # full under-range (empty) or full over-range (empty)
            if last_idx < 0 or first_idx is None:
                continue

            # part out of range (mixed and shorter as half before the first label's start)
            if first_idx < 0:
                if self._params['drop_mixed'] and self._params['drop_shorter']:
                    # goto next segment
                    continue
                else:
                    # cut to start
                    b = self._signal.get_start_time()

            # part out of range (shorter as half before the first label's start)
            if last_idx is None:
                if self._params['drop_shorter']:
                    # skip win == end (based on what next_times returns)
                    continue
                else:
                    # cut to end
                    e = self._signal.get_end_time()

            if not isinstance(self._labsig, _UnevenlySignal):
                break

            # labels segment bounds, can't be None
            first_iidx = self._labsig.get_iidx(b)
            last_iidx = self._labsig.get_iidx(e)

            # first label
            if first_iidx is not None:
                label = self._labsig[first_iidx]
            else:
                # None label
                first_iidx = -1

            # For each label in b,e from the second sample
            for i in range(first_iidx + 1, last_iidx):
                if label != self._labsig[i]:
                    # This is a mixed segment
                    if self._params['drop_mixed']:
                        # goto next segment
                        continue
                    else:
                        # keep
                        label = None
                    break
            break
        return b, e, label


class FixedSegments(_SegmentsWithLabelSignal):
    """
    Constant length (time) segments specifying segment step and segment width in seconds. A label signal from which to
    take labels can be specified.
    __init__(self, step, width=0, start=0, labels=None, drop_mixed=True)
    """

    def __init__(self, step, width=0, start=0, labels=None, drop_mixed=True, **kwargs):
        super(FixedSegments, self).__init__(step=step, width=width, start=start, labels=labels,
                                            drop_mixed=drop_mixed, **kwargs)
        assert labels is None or isinstance(labels, _UnevenlySignal),\
            "The parameter 'labels' should be an UnevenlySignal."
        self._step = None
        self._width = None
        self._t = None

    def init_segmentation(self):
        self._step = self._params["step"]
        w = self._params["width"]
        self._width = w if w > 0 else self._step
        self._labsig = self._params["labels"]
        s = self._params["start"]
        # TODO : we could also have signals with start_time < 0 ! -> s could be < 0 ==> if s > signal.get_start_time() else self._signal.get_start_time()
        self._t = s if s > 0 else self._signal.get_start_time()

    def next_times(self):
        b = self._t
        self._t += self._step
        e = b + self._width
        if b >= self._signal.get_end_time():
            raise StopIteration()
        return b, e


class CustomSegments(_SegmentsWithLabelSignal):
    """
    Custom begin-end time-segments.
    __init__(self, begins, ends, labels=None, drop_mixed=True)
    """

    def __init__(self, begins, ends, labels=None, drop_mixed=True, **kwargs):
        super(CustomSegments, self).__init__(begins=begins, ends=ends, labels=labels, drop_mixed=drop_mixed, **kwargs)
        assert len(begins) == len(ends), "The number of begins has to be equal to the number of ends :)"
        assert labels is None or isinstance(labels, _UnevenlySignal),\
            "The parameter 'labels' should be an UnevenlySignal."
        self._i = None
        self._b = None
        self._e = None

    def init_segmentation(self):
        self._i = -1
        self._b = self._params['begins']
        self._e = self._params['ends']

    def next_times(self):
        self._i += 1
        if self._i < len(self._b):
            return self._b[self._i], self._e[self._i]
        else:
            raise StopIteration()


# WAS: class ExistingSegments(SegmentsGenerator):


class LabelSegments(_SegmentsWithLabelSignal):
    """
    Generates a list of segments from a label signal, allowing to collapse subsequent equal labels.
    __init__(labels)
    """

    def __init__(self, labels, **kwargs):
        super(LabelSegments, self).__init__(labels=labels, **kwargs)
        assert labels is None or isinstance(labels, _UnevenlySignal),\
            "The parameter 'labels' should be an UnevenlySignal."
        self._i = None
        self._labsig = None

    def init_segmentation(self):
        self._i = 0
        self._labsig = self._params['labels']

    def next_times(self):
        b = self._labsig.get_time(self._i)
        e = self._labsig.get_time(self._i + 1)
        self._i += 1
        if b is None:
            raise StopIteration()
        return b, e
