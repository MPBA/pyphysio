# coding=utf-8
import numpy as _np
from ..Utility import PhUI as _PhUI, abstractmethod as _abstract
from ..BaseSegmentation import SegmentsGenerator, Segment
from ..Signal import Signal as _Signal

__author__ = 'AleB'


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

                if first == last:
                    last += 1

                lab_seg = self._labsig.segment_iidx(first, last)
                lab_first = lab_seg[0]

                if len(lab_seg) == 1 or (lab_seg[:1:-1] == lab_first).all(): ###[AB]
                    label = lab_first
                else:
                    if self._params['drop_mixed']:
                        continue
                    else:
                        label = None
            break

        return b, e, label


class RandomFixedSegments(_SegmentsWithLabelSignal):
    """
    Fixed length segments iterator, at random start timestamps, specifying step and width in seconds.

    A label signal from which to
    take labels can be specified.

    Parameters
    ----------
    width : float, >0
        time distance between subsequent segments.
    N : int, >0
        number of segments to be extracted.

    Optional parameters
    -------------------
    labels : array
        Signal of the labels
    drop_mixed : bool, default=True
        In case labels is specified, whether to drop segments with more than one label, if False the label of such
         segments is set to None.
    drop_cut : bool, default=True
        Whether to drop segments that are shorter due to the crossing of the signal end.
    """

    def __init__(self, N, width, start=None, stop = None, labels=None, drop_mixed=True, drop_cut=True, **kwargs):
        super(RandomFixedSegments, self).__init__(N=N, width=width, start=start, stop=stop, labels=labels, drop_cut=drop_cut,
                                            drop_mixed=drop_mixed, **kwargs)
        assert N > 0
        assert width > 0
        assert start is None or isinstance(start, float)
        assert stop is None or isinstance(stop, float)
        assert labels is None or isinstance(labels, _Signal),\
            "The parameter 'labels' should be a Signal."
        self._width = None
        self._tst = None
        self._tsp = None
        self._i = None

    def init_segmentation(self):
        self._N = self._params["N"]
        w = self._params["width"]
        self._width = w 
        self._labsig = self._params["labels"]
        self._tst = self._params["start"]
        self._tsp = self._params["stop"]
        
        #generate random start instants
        tst = self._signal.get_start_time() if self._tst is None else self._tst
        tsp = self._signal.get_end_time() if self._tsp is None else self._tsp
        tsp = tsp - w
        self._tst_randoms = _np.random.uniform(tst, tsp, self._N)
        
    def next_times(self):
        if self._i is None:
            self._i = 0
        
        if self._i < len(self._tst_randoms):
            b = self._tst_randoms[self._i]
            e = b + self._width
            self._i += 1
            return b, e
        else:
            raise StopIteration()
            

class FixedSegments(_SegmentsWithLabelSignal):
    """
    Fixed length segments iterator, specifying step and width in seconds.

    A label signal from which to
    take labels can be specified.

    Parameters
    ----------
    step : float, >0
        time distance between subsequent segments.

    Optional parameters
    -------------------
    width : float, >0, default=step
        time distance between subsequent segments.
    start : float
        start time of the first segment
    labels : array
        Signal of the labels
    drop_mixed : bool, default=True
        In case labels is specified, whether to drop segments with more than one label, if False the label of such
         segments is set to None.
    drop_cut : bool, default=True
        Whether to drop segments that are shorter due to the crossing of the signal end.
    """

    def __init__(self, step, width=None, start=None, labels=None, drop_mixed=True, drop_cut=True, **kwargs):
        super(FixedSegments, self).__init__(step=step, width=width, start=start, labels=labels, drop_cut=drop_cut,
                                            drop_mixed=drop_mixed, **kwargs)
        assert step > 0
        assert width is None or width > 0
        assert start is None or start > 0
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
    Custom segments iterator, specifying an array of begin times and an array of end times.

    Parameters
    ----------
    begins : array or list
        Array of the begin times of the segments to return.
    ends : array or list
        Array of the end times of the segments to return, of the same length of 'begins'.

    Optional parameters
    -------------------
    labels : array or list
        Signal of the labels
    drop_mixed : bool, default=True
        In case labels is specified, weather to drop segments with more than one label, if False the label of such
         segments is set to None.
    drop_cut : bool, default=True
        Weather to drop segments that are shorter due to the crossing of the signal end.
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
        self._labsig = self._params["labels"]

    def next_times(self):
        self._i += 1
        if self._i < len(self._b):
            return self._b[self._i], self._e[self._i]
        else:
            raise StopIteration()


class LabelSegments(_SegmentsWithLabelSignal):
    """
    Generates a list of segments from a label signal, allowing to collapse subsequent equal samples.

    Parameters
    ----------
    labels : array or list
        Signal of the labels

    Optional parameters
    -------------------
    drop_mixed : bool, default=True
        In case labels is specified, weather to drop segments with more than one label, if False the label of such
         segments is set to None.
    drop_cut : bool, default=True
        Weather to drop segments that are shorter due to the crossing of the signal end.
    """

    def __init__(self, labels, drop_mixed=True, drop_cut=True, **kwargs):
        super(LabelSegments, self).__init__(labels=labels, drop_mixed=drop_mixed, drop_cut=drop_cut, **kwargs)
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
