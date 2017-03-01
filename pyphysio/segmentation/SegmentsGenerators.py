# coding=utf-8
from ..Utility import PhUI as _PhUI
from ..BaseSegmentation import SegmentsGenerator, Segment
from ..Signal import Signal as _Signal
from pyphysio.Utility import abstractmethod as _abstract

__author__ = 'AleB'

# WAS: class LengthSegments(SegmentsGenerator):


class _SegmentsWithLabelSignal(SegmentsGenerator):
    # Assumed: label signal extended over the end by holding the value

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

    def check_drop_and_range(self, b, e):
        drop = True, None, None

        # signal segment bounds
        first_idx = self._signal.get_idx(b) if b is not None else None
        last_idx = self._signal.get_idx(e) if e is not None else None

        # full under-range (empty) or full over-range (empty)
        if last_idx < 0 or first_idx is None:
            return drop

        # part out of range: mixed and shorter (as partially before the first label's begin)
        if first_idx < 0:
            if self._params['drop_mixed'] or self._params['drop_shorter']:
                # goto next segment
                return drop
            else:
                # cut to start
                b = self._signal.get_start_time()

        # part out of range: shorter but not mixed (as half after the last label's end)
        if last_idx is None:
            if self._params['drop_shorter']:
                return drop
            else:
                # cut to end
                e = self._signal.get_end_time()

        return False, b, e

    def next_segment_mix_labels(self):
        label = b = e = None
        while True:
            # break    ==> keep
            # continue ==> drop

            b, e = self.next_times()

            drop, b, e = self.check_drop_and_range(b, e)

            if drop:
                continue

            if not isinstance(self._labsig, _Signal):
                label = None
            else:
                # labels segment bounds, can't be None
                first = self._labsig.get_iidx(b)
                last = self._labsig.get_iidx(e)

                # First label
                label = self._labsig[first]

                # Check if classically mixed
                # compare with first each label in [b+1, e)
                for i in range(first + 1, last):
                    if label != self._labsig[i]:
                        # This is a mixed segment
                        first = None
                        break  # for

                if first is None:
                    # mixed window
                    if self._params['drop_mixed']:
                        # goto next segment
                        continue
                    else:
                        # keep with first label (i<0) == None
                        label = None

            # keep
            break

        return b, e, label


class FixedSegments(_SegmentsWithLabelSignal):
    """
    Constant length (time) segments specifying segment step and segment width in seconds. A label signal from which to
    take labels can be specified.
    __init__(self, step, width=0, start=0, labels=None, drop_mixed=True)
    """

    def __init__(self, step, width=None, start=None, labels=None, drop_mixed=True, **kwargs):
        super(FixedSegments, self).__init__(step=step, width=width, start=start, labels=labels,
                                            drop_mixed=drop_mixed, **kwargs)
        assert labels is None or isinstance(labels, _Signal),\
            "The parameter 'labels' should be a Signal."
        self._step = None
        self._width = None
        self._t = None

    def init_segmentation(self):
        self._step = self._params["step"]
        w = self._params["width"]
        self._width = w if w is not None else self._step
        self._labsig = self._params["labels"]
        s = self._params["start"]
        self._t = s if s is not None else self._signal.get_start_time()

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
        assert labels is None or isinstance(labels, _Signal),\
            "The parameter 'labels' should be an Signal."
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
        assert labels is None or isinstance(labels, _Signal),\
            "The parameter 'labels' should be an Signal."
        self._i = None
        self._labsig = None

    def init_segmentation(self):
        self._i = 0
        self._labsig = self._params['labels']

    def next_times(self):
        i = self._i
        b = self._labsig.get_time_from_iidx(self._i)
        if b is None:
            raise StopIteration()
        while i < len(self._labsig) and self._labsig[self._i] == self._labsig[i]:
            i += 1
        self._i = i
        e = self._labsig.get_time_from_iidx(self._i)
        return b, e
