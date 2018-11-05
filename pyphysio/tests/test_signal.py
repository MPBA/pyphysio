# coding=utf-8
from __future__ import division

from . import ph, TestData, approx
import numpy as np

import pytest

__author__ = 'aleb'


# noinspection PyAttributeOutsideInit
class TestSignal(object):
    def setup_class(self):
        self.samples = 1000
        self.freq_down = 256
        self.freq1 = 100
        self.freq2 = 1024
        self.start1 = 3
        self.start2 = 13370003
        self.nature = "una_bif-fa"

        np.random.seed(1234)
        self.x_vals = np.cumsum(np.random.rand(1, self.samples) * 9 + 1).astype(int)
        self.y_vals = np.cumsum(np.random.rand(1, self.samples) - .5) * 100

        self.s = ph.EvenlySignal(values=TestData.ecg(),
                                 sampling_freq=self.freq1,
                                 signal_nature=self.nature,
                                 start_time=self.start1,
                                 )

        self.empty_s = ph.EvenlySignal(values=[],
                                       sampling_freq=self.freq1,
                                       signal_nature=self.nature,
                                       start_time=self.start1,
                                       )

        self.us = ph.UnevenlySignal(values=self.y_vals,
                                    x_values=self.x_vals,
                                    sampling_freq=self.freq1,
                                    signal_nature=self.nature,
                                    start_time=self.start1,
                                    x_type='indices'
                                    )

    def test_signal_instantiation(self):
        # instants
        ust = ph.UnevenlySignal(values=self.y_vals,
                                x_values=self.x_vals / self.freq1 + self.start1,
                                sampling_freq=self.freq1,
                                signal_nature=self.nature,
                                start_time=self.start1,
                                x_type='instants'
                                )

        assert len(ust) == len(self.us)
        assert len(ust.get_indices()) == len(self.x_vals)  # FIXME early tests for get_indices()

        for i, (x, y) in enumerate(zip(ust.get_indices(), self.x_vals)):
            assert x == y, "index %d" % i

        # start_time on first time
        ph.UnevenlySignal([1, 2, 3],
                          x_values=[1, 2, 3],
                          x_type='instants',
                          sampling_freq=10,
                          start_time=1,
                          )

        # start_time negative
        ph.UnevenlySignal([1, 2, 3],
                          x_values=[1, 2, 3],
                          x_type='instants',
                          sampling_freq=10,
                          start_time=-1.1,
                          )

        # start_time after first time
        with pytest.raises(AssertionError):
            ph.UnevenlySignal([1, 2, 3],
                              x_values=[1, 2, 3],
                              x_type='instants',
                              sampling_freq=10,
                              start_time=1.1,
                              )

        # start_time not numeric
        with pytest.raises(AssertionError):
            ph.EvenlySignal([], 1024, start_time='ugo')

        # None values
        with pytest.raises(AssertionError):
            ph.EvenlySignal(None, 1024)

        # non 1-dim values 1
        with pytest.raises(AssertionError):
            ph.EvenlySignal([[1], [2]], 1024)

        # non 1-dim values 2
        with pytest.raises(AssertionError):
            ph.EvenlySignal([[1, 2]], 1024)

        # x_values missing
        with pytest.raises(AssertionError):
            ph.UnevenlySignal([])

        # x_values not strictly monotonic
        with pytest.raises(AssertionError):
            ph.UnevenlySignal(range(3), x_values=[0, 1.0, 1])

        # x_values length mismatch
        with pytest.raises(AssertionError):
            ph.UnevenlySignal(range(3), x_values=range(4))

        # x_type unknown
        with pytest.raises(AssertionError):
            ph.UnevenlySignal(range(3), x_values=range(3), x_type='bart')

    def test_signal_base(self):
        # ph
        assert isinstance(self.s.ph, dict)
        assert len(self.s.ph) >= 3

        # get_start_time
        assert self.s.get_start_time() == self.start1

        # set_start_time
        self.s.set_start_time(self.start2)
        assert self.s.get_start_time() == self.start2

        # get_sampling_freq
        assert self.s.get_sampling_freq() == self.freq1

        # set_sampling_freq
        self.s.set_sampling_freq(self.freq2)
        assert self.s.get_sampling_freq() == self.freq2

        # EvenlySignal#get_end_time
        assert self.s.get_end_time() == self.s.get_start_time() + len(self.s) / self.s.get_sampling_freq()  # future div

        # UnevenlySignal#get_end_time
        assert self.us.get_end_time() == self.us.get_start_time() + (self.x_vals[-1] + 1.) / self.us.get_sampling_freq()

        # get_duration
        assert isinstance(self.s.get_duration(), float)
        assert self.s.get_duration() == self.s.get_end_time() - self.s.get_start_time()

        # get_idx
        assert self.s.get_idx(self.s.get_start_time()) == 0
        assert self.s.get_idx(self.s.get_end_time()) is not None
        assert self.s.get_idx(10) == (10 - self.s.get_start_time()) * self.s.get_sampling_freq()

        # plot
        import matplotlib.lines
        a = self.s.plot()
        assert isinstance(a, list)
        assert isinstance(a[0], matplotlib.lines.Line2D)
        import matplotlib.collections
        a = self.s.plot("ro")
        assert isinstance(a, list)
        assert isinstance(a[0], matplotlib.lines.Line2D)
        assert isinstance(self.s.plot("|r"), matplotlib.collections.LineCollection)

        # pickleability
        ps = ph.Signal.from_pickleable(self.s.pickleable)
        assert ps.get_start_time() == self.s.get_start_time()
        assert ps.get_sampling_freq() == self.s.get_sampling_freq()
        assert ps.get_end_time() == self.s.get_end_time()
        assert ps.get_signal_nature() == self.s.get_signal_nature()
        assert len(ps) == len(self.s)

    def test_evenly_signal_base(self):
        # get_times
        t = self.s.get_times()
        assert len(self.s.get_values()) == len(t)  # length
        assert len(np.where(np.diff(t) <= 0)[0]) == 0  # strong monotonicity

        # get_time
        assert self.s.get_time(0) == self.s.get_start_time()
        assert self.s.get_time(len(t) - 1) == approx(self.s.get_end_time(), rel=.00002)
        assert self.s.get_time(len(t)) is not None
        assert self.s.get_time(10) == self.s.get_start_time() + 10 / self.s.get_sampling_freq()

        # get_iidx
        assert self.s.get_iidx(0) == self.s.get_idx(0)
        assert self.s.get_iidx(len(t) - 1) == self.s.get_idx(len(t) - 1)
        assert self.s.get_iidx(len(t)) == self.s.get_idx(len(t))

        # get_time_from_iidx
        assert self.s.get_time_from_iidx(0) == self.s.get_time(0)
        assert self.s.get_time_from_iidx(len(t) - 1) == self.s.get_time(len(t) - 1)
        assert self.s.get_time_from_iidx(len(t)) == self.s.get_time(len(t))

    def test_unevenly_signal_base(self):
        # get_indices
        for i, (a, b) in enumerate(zip(self.us.get_indices(), self.x_vals)):
            assert a == b

        # get_times
        for i, (a, b) in enumerate(zip(self.us.get_times(),
                                       self.x_vals / self.us.get_sampling_freq() + self.us.get_start_time())):
            assert a == b, "Index %d" % i

        # get_time
        assert self.us.get_time(0) == self.us.get_start_time()
        assert self.us.get_time_from_iidx(len(self.us) - 1) + 1. / self.us.get_sampling_freq() == self.us.get_end_time()
        assert self.us.get_time_from_iidx(len(self.us)) == self.us.get_time_from_iidx(len(self.us) - 1)
        assert self.us.get_time(10) == self.us.get_start_time() + 10 / self.us.get_sampling_freq()

        #
