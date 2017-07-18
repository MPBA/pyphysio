# coding=utf-8
from __future__ import division

from . import ph, TestData
from math import ceil as _ceil
import numpy as np

import unittest

__author__ = 'aleb'


# noinspection PyArgumentEqualDefault
class GeneralTest(unittest.TestCase):
    def test_fmap_and_algo(self):
        s = ph.EvenlySignal((np.sin(np.random.rand(100) * 3.14 - (3.14 / 2)) + 1) * 93, 15)
        lf = ph.PowerInBand(freq_max=1, freq_min=0.001, method='ar')
        hf = ph.PowerInBand(freq_max=4, freq_min=1, method='ar')

        algos = [
            ph.Mean(),
            ph.StDev(),
            ph.NNx(threshold=100),
            ph.PowerInBand(interp_freq=20, freq_max=4, freq_min=0.001, method='ar'),
            ph.PowerInBand(interp_freq=20, freq_min=4, freq_max=15, method='ar'),
            ph.algo(lambda d, p: lf(d) / hf(d))(),
            ph.algo(lambda d, p: len(d))()
        ]

        g = ph.FixedSegments(step=1, width=1.5)
        l = len([x for x in g(s)])
        r, ignored = ph.fmap(g(s), algos)
        n = 3 + len(algos)
        for i in r:
            self.assertEquals(len(i), n)

        self.assertEquals(len(r), l)

    def test_fmap(self):
        samples = 1000
        freq_down = 13
        freq = freq_down * 7
        start = 1460713373
        nature = "una_bif-fa"
        np.random.seed(1234)

        s = ph.EvenlySignal(values=np.cumsum(np.random.rand(1, samples) - .5) * 100,
                            sampling_freq=freq,
                            signal_nature=nature,
                            start_time=start,
                            )

        features = [
            ph.PNNx(threshold=50),
            ph.PNNx(threshold=25),
            ph.PNNx(threshold=10),
            ph.PNNx(threshold=5),
            ph.Mean(),
            ph.StDev(),
            ph.Median(),
        ]

        segmenter = ph.FixedSegments(width=3, step=2)
        segments = [i for i in segmenter(s)]

        results, columns = ph.fmap(segments, features, s)

        self.assertEqual(results.shape, (len(segments), len(features) + 3))
        self.assertEqual(len(columns), len(features) + 3)

    def test_ex_more(self):
        s = ph.EvenlySignal(np.cumsum(np.random.rand(1000) - .5) * 100, 10)

        w1 = ph.FixedSegments(step=2, width=3)(s)
        w3 = ph.LabelSegments(labels=ph.UnevenlySignal(
            values=['a', 'a', 'b', 'a', 'r', 's', 'r', 'b'],
            x_values=[10, 12, 13.5, 14.3, 15.6, 20.1123, 25, 36.8],
            sampling_freq=10,
            start_time=8,
            x_type='instants'))(s)
        w3i = [x for x in w3]
        w4 = ph.CustomSegments(begins=[x.get_begin_time() for x in w3i],
                               ends=[x.get_end_time() for x in w3i])(s)
        w4i = [x for x in w4]

        self.assertEqual(len(w4i), len(w3i))

        y1 = [x for x in w1]
        y3 = [x for x in w3]
        ph.fmap(w1, [ph.Mean(), ph.StDev(), ph.NNx(threshold=31)])
        ph.fmap(w3, [ph.Mean(), ph.StDev(), ph.NNx(threshold=31)])
        ph.fmap(y1, [ph.Mean(), ph.StDev(), ph.NNx(threshold=31)])
        ph.fmap(y3, [ph.Mean(), ph.StDev(), ph.NNx(threshold=31)])

        # noinspection PyArgumentEqualDefault
        sd2 = s.resample(1, 'linear')
        sd3 = s.resample(1, 'nearest')
        sd4 = s.resample(1, 'zero')
        sd5 = s.resample(1, 'slinear')
        sd6 = s.resample(1, 'quadratic')
        sd7 = s.resample(1, 'cubic')

        self.assertEqual(len(sd2), 100)
        self.assertEqual(len(sd3), 100)
        self.assertEqual(len(sd4), 100)
        self.assertEqual(len(sd5), 100)
        self.assertEqual(len(sd6), 100)
        self.assertEqual(len(sd7), 100)

        # noinspection PyArgumentEqualDefault
        so2 = s.resample(20, 'linear')
        so3 = s.resample(20, 'nearest')
        so4 = s.resample(20, 'zero')
        so5 = s.resample(20, 'slinear')
        so6 = s.resample(20, 'quadratic')
        so7 = s.resample(20, 'cubic')

        self.assertEqual(len(so2), 2000)
        self.assertEqual(len(so3), 2000)
        self.assertEqual(len(so4), 2000)
        self.assertEqual(len(so5), 2000)
        self.assertEqual(len(so6), 2000)
        self.assertEqual(len(so7), 2000)

        so1 = s.resample(21)

        self.assertEqual(len(so1), 2100)

    def test_evenly_signal_base(self):
        samples = 1000
        freq_down = 13
        freq = freq_down * 7
        start = 1460713373
        nature = "una_bif-fa"

        s = ph.EvenlySignal(values=np.cumsum(np.random.rand(1, samples) - .5) * 100,
                            sampling_freq=freq,
                            signal_nature=nature,
                            start_time=start,
                            )

        assert s.get_sampling_freq() == freq
        assert s.get_signal_nature() == nature
        assert s.get_start_time() == start

        # ineffective
        s.resample(freq_down)

        # FIXME: this is part of test_evenly_signal_resample (?)
        # assert properties of resampled s

        # length Y
        self.assertEqual(len(s), samples)
        self.assertEqual(len(s.get_values()), samples)
        # length X
        self.assertEqual(len(s.get_times()), samples)
        # sampling frequency in Hz
        self.assertEqual(s.get_sampling_freq(), freq)
        # duration in seconds
        self.assertAlmostEqual(s.get_duration(), samples / freq, 6)
        # start timestamp
        self.assertEqual(s.get_start_time(), start)
        # end timestamp
        self.assertEqual(s.get_end_time(), start + s.get_duration())
        # start time
        self.assertEqual(s.get_signal_nature(), nature)

    # TODO : test_unevenly_signal_base with x_values of type 'indices'
    def test_unevenly_signal_base(self):
        samples = 200
        freq = 13
        start = 0
        nature = "una_bif-fa"

        s = ph.UnevenlySignal(values=np.cumsum(np.random.rand(1, samples) - .5) * 100,
                              x_values=np.cumsum(np.random.rand(1, samples)) * 100 + start,
                              sampling_freq=freq,
                              signal_nature=nature,
                              start_time=start,
                              x_type='indices'
                              )
        # assert properties of original s
        #
        # length Y
        self.assertEqual(len(s), samples)
        self.assertEqual(len(s.get_values()), samples)
        # length X
        self.assertEqual(len(s.get_indices()), samples)
        # sampling frequency in Hz
        self.assertEqual(s.get_sampling_freq(), freq)
        # start timestamp
        self.assertEqual(s.get_start_time(), start)
        # end timestamp
        self.assertEqual(s.get_end_time(), start + s.get_duration())
        # start time
        self.assertEqual(s.get_signal_nature(), nature)

    def test_evenly_signal_resample(self):
        samples = 1000
        freq_down = 7
        freq = freq_down * 13
        freq_up = freq * 3
        freq_down_r = freq / 5
        start = 1460713373
        nature = "una_bif-fa"

        s = ph.EvenlySignal(values=np.cumsum(np.random.rand(1, samples) - .5) * 100,
                            sampling_freq=freq,
                            signal_nature=nature,
                            start_time=start,
                            )

        # TODO: test also different interpolationi methods
        def check_resampled(freq_new):
            # resample
            resampled = s.resample(freq_new)
            # new number of samples
            new_samples = _ceil(samples * freq_new / freq)

            # length Y
            self.assertEqual(len(resampled), new_samples)
            # length X
            self.assertEqual(len(resampled.get_times()), len(resampled))
            # sampling frequency in Hz
            self.assertEqual(resampled.get_sampling_freq(), freq_new)
            # duration differs less than 1s from the original
            self.assertLess(abs(resampled.get_duration() - samples / freq), 1)
            # start timestamp
            self.assertEqual(resampled.get_start_time(), start)
            # end timestamp
            self.assertEqual(resampled.get_end_time(), start + resampled.get_duration())
            # start time
            self.assertEqual(resampled.get_signal_nature(), nature)

        # down-sampling int
        check_resampled(freq_down)
        # over-sampling int
        check_resampled(freq_up)
        # down-sampling rationale
        check_resampled(freq_down_r)

    def test_unevenly_signal_to_evenly(self):
        samples = 200
        indexes = np.cumsum(np.round(np.random.rand(1, samples) + 1))
        freq = 13
        start = 0
        nature = "una_bif-fa"

        s = ph.UnevenlySignal(values=np.cumsum(np.random.rand(1, samples) - .5) * 100,
                              x_values=indexes,
                              sampling_freq=freq,
                              signal_nature=nature,
                              start_time=start,
                              x_type='indices'
                              )

        # conversion
        s = s.to_evenly()

        # length
        self.assertEqual(len(s.get_values()), len(s))
        # sampling frequency in Hz
        self.assertEqual(s.get_sampling_freq(), freq)
        # start timestamp
        self.assertEqual(s.get_start_time(), s.get_time_from_iidx(0))
        # end timestamp
        self.assertEqual(s.get_end_time(), s.get_time_from_iidx(0) + s.get_duration())
        # start time
        self.assertEqual(s.get_signal_nature(), nature)

    def test_segmentation(self):
        samples = 1000
        freq_down = 7
        freq = freq_down * 13
        start = 1460000000
        nature = "una_bif-fa"

        s = ph.EvenlySignal(values=np.cumsum(np.random.rand(1, samples) - .5) * 100,
                            sampling_freq=freq,
                            signal_nature=nature,
                            start_time=start
                            )

        def function(y):
            for j in range(len(y)):
                self.assertLessEqual(y[j].get_begin(), y[j].get_end())
                self.assertGreaterEqual(y[j].get_begin_time(), s.get_start_time())
                self.assertGreaterEqual(y[j].get_end_time(), s.get_start_time())
                # # list[x:y] where y > len(list) is a valid usage in python so next lines are cut
                # self.assertLessEqual(y[j].get_begin(), len(s))
                # self.assertLessEqual(y[j].get_end(), len(s))

        w1 = ph.FixedSegments(step=2, width=3)(s)
        y1 = [x for x in w1]
        function(y1)

        w2 = ph.FixedSegments(step=100, width=121)(s)
        y2 = [x for x in w2]
        function(y2)

        w3 = ph.LabelSegments(labels=ph.UnevenlySignal(values=[0, 0, 1, 0, 2, 3, 2, 1],
                                                       x_values=np.array(
                                                           [10, 12, 13.5, 14.3, 15.6, 20.1123, 25, 36.8]) + start,
                                                       # start_time=start,
                                                       x_type='instants'
                                                       ))(s)

        y3 = [x for x in w3]
        function(y3)

        w4 = ph.CustomSegments(begins=np.arange(s.get_start_time(), s.get_end_time(), 3),
                               ends=np.arange(s.get_start_time() + 1, s.get_end_time(), 3))(s)
        y4 = [x for x in w4]
        function(y4)

        r = [ph.fmap(w1, [ph.Mean(), ph.StDev(), ph.NNx(threshold=31)]),
             ph.fmap(w2, [ph.Mean(), ph.StDev(), ph.NNx(threshold=31)]),
             ph.fmap(w3, [ph.Mean(), ph.StDev(), ph.NNx(threshold=31)]),
             ph.fmap(w4, [ph.Mean(), ph.StDev(), ph.NNx(threshold=31)]),
             ph.fmap(y1, [ph.Mean(), ph.StDev(), ph.NNx(threshold=31)]),
             ph.fmap(y2, [ph.Mean(), ph.StDev(), ph.NNx(threshold=31)]),
             ph.fmap(y3, [ph.Mean(), ph.StDev(), ph.NNx(threshold=31)]),
             ph.fmap(y4, [ph.Mean(), ph.StDev(), ph.NNx(threshold=31)])]

        n = 4
        for i in range(n):
            self.assertEqual(len(r[i]), len(r[i + n]))

    def test_cache_with_time_domain(self):
        samples = 1000
        freq_down = 13
        freq = freq_down * 7
        start = 1460713373
        nature = "una_bif-fa"

        s = ph.EvenlySignal(values=np.cumsum(np.random.rand(1, samples) - .5) * 100,
                            sampling_freq=freq,
                            signal_nature=nature,
                            start_time=start,
                            )

        def function():
            # Mean
            self.assertEqual(np.nanmean(s), ph.Mean()(s))
            # Min
            self.assertEqual(np.nanmin(s), ph.Min()(s))
            # Max
            self.assertEqual(np.nanmax(s), ph.Max()(s))
            # Range
            self.assertEqual(np.nanmax(s) - np.nanmin(s), ph.Range()(s))
            # Median
            self.assertEqual(np.median(s), ph.Median()(s))
            # StDev
            self.assertEqual(np.nanstd(s), ph.StDev()(s))

        # raw
        function()
        # cache
        function()

    def test_evenly_slicing(self):
        samples = 1000
        x1 = 200
        x2 = 201
        x3 = 590
        x4 = 999
        freq_down = 7
        freq = freq_down * 13
        start = 1460713373
        nature = "una_bif-fa"

        s = ph.EvenlySignal(values=np.cumsum(np.random.rand(1, samples) - .5) * 100,
                            sampling_freq=freq,
                            signal_nature=nature,
                            start_time=start,
                            )

        s1 = s.segment_idx(0, x4)
        s2 = s.segment_idx(x1, x3)
        s3 = s.segment_idx(x2, -1)
        s4 = s.segment_idx(x3, samples)
        s5 = s.segment_idx(0, None)
        s6 = s.segment_idx(None, x2)
        s7 = s.segment_idx(None, None)

        self.assertEqual(len(s1), x4 - 0)
        self.assertEqual(len(s2), x3 - x1)
        self.assertEqual(len(s3), samples - x2 - 1)
        self.assertEqual(len(s4), samples - x3)
        self.assertEqual(len(s5), samples)
        self.assertEqual(len(s6), x2)
        self.assertEqual(len(s7), samples)
        self.assertEqual(s1.get_start_time(), s.get_time(0))
        self.assertEqual(s2.get_start_time(), s.get_time(x1))
        self.assertEqual(s3.get_start_time(), s.get_time(x2))
        self.assertEqual(s4.get_start_time(), s.get_time(x3))
        self.assertEqual(s5.get_start_time(), s.get_time(0))
        self.assertEqual(s6.get_start_time(), s.get_time(0))
        self.assertEqual(s7.get_start_time(), s.get_time(0))
        self.assertEqual(s1.get_end_time(), s1.get_time(x4 - 0 - 1) + 1. / s1.get_sampling_freq())
        self.assertEqual(s2.get_end_time(), s2.get_time(x3 - x1 - 1) + 1. / s2.get_sampling_freq())
        self.assertEqual(s3.get_end_time(), s3.get_time(samples - x2 - 1 - 1) + 1. / s3.get_sampling_freq())
        self.assertEqual(s4.get_end_time(), s4.get_time(samples - x3 - 1) + 1. / s4.get_sampling_freq())
        self.assertEqual(s5.get_end_time(), s5.get_time(samples - 1) + 1. / s5.get_sampling_freq())
        self.assertEqual(s6.get_end_time(), s6.get_time(x2 - 1) + 1. / s6.get_sampling_freq())
        self.assertEqual(s7.get_end_time(), s7.get_time(samples - 1) + 1. / s7.get_sampling_freq())

    def test_unevenly_slicing(self):
        samples = 1000
        x1 = 200
        x2 = 201
        x3 = 590
        x4 = 999
        freq = 7 * 13
        start = 0
        nature = "una_bif-fa"

        x_vals = np.cumsum(np.random.rand(1, samples) * 9 + 1).astype(int)

        s = ph.UnevenlySignal(values=np.cumsum(np.random.rand(1, samples) - .5) * 100,
                              x_values=x_vals,
                              sampling_freq=freq,
                              signal_nature=nature,
                              start_time=start,
                              x_type='indices'
                              )

        s1 = s.segment_iidx(0, x4)
        s2 = s.segment_iidx(x1, x3)
        s3 = s.segment_iidx(x2, -1)
        s4 = s.segment_iidx(x3, samples)
        s5 = s.segment_iidx(0, None)
        s6 = s.segment_iidx(None, x2)
        s7 = s.segment_iidx(None, None)

        self.assertEqual(len(s1), x4 - 0)
        self.assertEqual(len(s2), x3 - x1)
        self.assertEqual(len(s3), samples - x2 - 1)
        self.assertEqual(len(s4), samples - x3)
        self.assertEqual(len(s5), samples)
        self.assertEqual(len(s6), x2)
        self.assertEqual(len(s7), samples)

        self.assertEqual(len(s1), len(s1.get_indices()))
        self.assertEqual(len(s2), len(s2.get_indices()))
        self.assertEqual(len(s3), len(s3.get_indices()))
        self.assertEqual(len(s4), len(s4.get_indices()))
        self.assertEqual(len(s5), len(s5.get_indices()))
        self.assertEqual(len(s6), len(s6.get_indices()))
        self.assertEqual(len(s7), len(s7.get_indices()))

        self.assertEqual(s1.get_indices()[0], 0)
        self.assertEqual(s2.get_indices()[0], 0)
        self.assertEqual(s3.get_indices()[0], 0)
        self.assertEqual(s4.get_indices()[0], 0)
        self.assertEqual(s5.get_indices()[0], 0)
        self.assertEqual(s6.get_indices()[0], 0)
        self.assertEqual(s7.get_indices()[0], 0)
        self.assertEqual(s1.get_indices()[-1], s.get_indices()[x4 - 1] - s.get_indices()[0])
        self.assertEqual(s2.get_indices()[-1], s.get_indices()[x3 - 1] - s.get_indices()[x1])
        self.assertEqual(s3.get_indices()[-1], s.get_indices()[-1 - 1] - s.get_indices()[x2])
        self.assertEqual(s4.get_indices()[-1], s.get_indices()[samples - 1] - s.get_indices()[x3])
        self.assertEqual(s5.get_indices()[-1], s.get_indices()[samples - 1] - s.get_indices()[0])
        self.assertEqual(s6.get_indices()[-1], s.get_indices()[x2 - 1] - s.get_indices()[0])
        self.assertEqual(s7.get_indices()[-1], s.get_indices()[samples - 1] - s.get_indices()[0])

    # noinspection PyMethodMayBeStatic
    def test_signal_plot(self):
        e = ph.EvenlySignal(values=TestData.ecg()[:10000], sampling_freq=1024, signal_nature="ecg")
        e, ignored, ignored, ignored = ph.PeakDetection(delta=1)(e)
        e = ph.UnevenlySignal(values=e, x_values=e, x_type='indices', sampling_freq=1024, signal_nature="ibi")
        e.plot("|b")
        e = ph.EvenlySignal(values=TestData.eda()[:10000], sampling_freq=1024, signal_nature="gsr")
        e.plot()
        e = ph.EvenlySignal(values=TestData.bvp()[:10000], sampling_freq=1024, signal_nature="bvp")
        e.plot()
        e = ph.EvenlySignal(values=TestData.resp()[:10000], sampling_freq=1024, signal_nature="resp")
        e.plot()

        # import matplotlib.pyplot as plt
        # plt.show()
        # plt.close('all')


if __name__ == '__main__':
    unittest.main()
