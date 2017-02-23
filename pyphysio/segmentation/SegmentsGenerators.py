# coding=utf-8
from ..Utility import PhUI as _PhUI
from ..BaseSegmentation import SegmentsGenerator, Segment
from ..Signal import Signal as _Signal
from pyphysio.Utility import abstractmethod as _abstract
from numpy import nan as _nan

__author__ = 'AleB'


# WAS: class LengthSegments(SegmentsGenerator):

class _SegmentsWithLabelSignal(SegmentsGenerator):

    def __init__(self, **kwargs):
        super(_SegmentsWithLabelSignal, self).__init__(**kwargs)
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
        while label is None:
            b, e = self.next_times()

            # signal segment bounds
            first_idx = self._signal.get_idx(b)
            last_idx = self._signal.get_idx(e)

            # in-range check b,e
            if first_idx is None:
                # out of range
                raise StopIteration()
            elif last_idx is None:
                # half out of range, cut to last idx
                e = self._signal.get_end_time()

            if not isinstance(self._labsig, _Signal):
                break

            # labels segment bounds
            first_idx = self._labsig.get_idx(b)
            last_idx = self._labsig.get_idx(e)

            if first_idx is None:
                label = self._labsig[-1]
            else:
                # first label
                first = self._labsig[first_idx]

                if last_idx is None:
                    last_idx = self._labsig.get_end_time()

                # For each label in b,e from the second sample
                for i in range(first_idx + 1, last_idx):
                    if first != self._labsig[i]:
                        # This is a mixed segment
                        if self._params['drop_mixed']:
                            # goto next segment
                            label = None
                        else:
                            # keep
                            label = _nan
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
        assert labels is None or isinstance(labels, _Signal), "Parameter labels must be a Signal"
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
        return b, e


class CustomSegments(_SegmentsWithLabelSignal):
    """
    Custom begin-end time-segments.
    __init__(self, begins, ends, labels=None, drop_mixed=True)
    """

    def __init__(self, begins, ends, labels=None, drop_mixed=True, **kwargs):
        super(CustomSegments, self).__init__(begins=begins, ends=ends, labels=labels, drop_mixed=drop_mixed, **kwargs)
        assert len(begins) == len(ends), "The number of begins has to be equal to the number of ends :)"
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
    __init__(labels, collapse=True)
    """

    def __init__(self, labels, collapse=False, **kwargs):
        super(LabelSegments, self).__init__(labels=labels, collapse=collapse, **kwargs)
        assert isinstance(labels, _Signal), "The parameter 'labels' should be a Signal."
        self._i = None
        self._labsig = None
        self._collapse = collapse

    def init_segmentation(self):
        self._i = 0
        self._labsig = self._params['labels']

    def next_times(self):
        if self._collapse:
            bi = self._i
            b = self._labsig.get_time(self._i)
            self._i += 1
            while self._labsig[bi] == self._labsig[self._i]:
                self._i += 1
            e = self._labsig.get_time(self._i + 1)
        else:
            b = self._labsig.get_time(self._i)
            e = self._labsig.get_time(self._i + 1)
            self._i += 1
        return b, e
