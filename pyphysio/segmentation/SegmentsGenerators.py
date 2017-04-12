# coding=utf-8
from ..Utility import PhUI as _PhUI
from ..BaseSegmentation import SegmentsGenerator, Segment
from ..Signal import Signal as _Signal
from pyphysio.Utility import abstractmethod as _abstract

__author__ = 'AleB'

# WAS: class LengthSegments(SegmentsGenerator):


class _SegmentsWithLabelSignal(SegmentsGenerator):
    # Assumed: label signal extended over the end by holding the value

    def __init__(self, drop_cut=True, drop_mixed=True, **kwargs):
        super(_SegmentsWithLabelSignal, self).__init__(drop_cut=drop_cut, drop_mixed=drop_mixed, **kwargs)
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

    def check_drop_and_range(self, s, b, e):
        drop = True, None, None

        # signal segment bounds

        # full under-range (empty) or full over-range (empty)
        if e < s.get_start_time() or b >= s.get_end_time():
            return drop

        # part before start: mixed and shorter (as partially before the first label's begin)
        if b < s.get_start_time():
            if self._params['drop_mixed'] or self._params['drop_cut']:
                # goto next segment (drop)
                return drop
            else:
                # cut to start
                b = s.get_start_time()

        # part after end: shorter but not mixed (as half after the last label's end)
        if e > s.get_end_time():
            if self._params['drop_cut']:
                # goto next segment (drop)
                return drop
            else:
                # cut to start
                e = s.get_end_time()

        # Don't drop, inclusive begin time, exclusive end time
        return False, b, e

    def next_segment_mix_labels(self):
        label = b = e = None
        while True:
            # break    ==> keep
            # continue ==> drop

            b, e = self.next_times()

            drop, b, e = self.check_drop_and_range(self._signal, b, e)

            if drop:
                continue

            if not isinstance(self._labsig, _Signal):
                label = None
            else:

                drop, _, _ = self.check_drop_and_range(self._labsig, b, e)

                if drop:
                    # partially or completely out of labsig range and have to drop it
                    label = None
                    continue

                # labels segment bounds, may be < 0 (None < 0)
                first = self._labsig.get_iidx(b)
                last = self._labsig.get_iidx(e)

                # first label
                label = self._labsig[first]

                # Check if classically mixed
                # compare with first each label in [b+1, e)
                for i in range(last - 1, first, -1):
                    if label != self._labsig[i]:
                        # this is a mixed segment
                        if self._params['drop_mixed']:
                            # goto next segment
                            continue
                        else:
                            # keep with label == None
                            label = None
                        break  # for
            # keep
            break

        return b, e, label


class FixedSegments(_SegmentsWithLabelSignal):
    """
    Constant length (time) segments specifying segment step and segment width in seconds. A label signal from which to
    take labels can be specified.
    __init__(self, step, width=0, start=0, labels=None, drop_mixed=True)
    """

    def __init__(self, step, width=None, start=None, labels=None, drop_mixed=True, drop_cut=True, **kwargs):
        super(FixedSegments, self).__init__(step=step, width=width, start=start, labels=labels, drop_cut=drop_cut,
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
        self._t = self._params["start"]

    def next_times(self):
        if self._t is None:
            self._t = self._signal.get_start_time()
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

    def __init__(self, begins, ends, labels=None, drop_mixed=True, drop_cut=True, **kwargs):
        super(CustomSegments, self).__init__(begins=begins, ends=ends, labels=labels, drop_cut=drop_cut,
                                             drop_mixed=drop_mixed, **kwargs)
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

    def __init__(self, labels, drop_mixed=True, drop_cut=True, **kwargs):
        super(LabelSegments, self).__init__(labels=labels, drop_cut=drop_cut, drop_mixed=drop_mixed, **kwargs)
        assert labels is None or isinstance(labels, _Signal),\
            "The parameter 'labels' should be an Signal."
        self._i = None
        self._labsig = None

    def init_segmentation(self):
        self._i = 0
        self._labsig = self._params['labels']

    def next_times(self):
        if self._i >= len(self._labsig):
            raise StopIteration()
        end = self._i
        while end < len(self._labsig) and self._labsig[self._i] == self._labsig[end]:
            end += 1
        b = self._labsig.get_time_from_iidx(self._i)
        e = self._labsig.get_time_from_iidx(end)
        self._i = end
        return b, e
