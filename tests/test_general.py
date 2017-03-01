# coding=utf-8
from __future__ import division

from context import ph
from math import ceil as _ceil
import numpy as np

import unittest

__author__ = 'aleb'


# noinspection PyArgumentEqualDefault
class GeneralTest(unittest.TestCase):
    def test_evenly_signal_base(self):
        samples = 1000
        freq_down = 13
        freq = freq_down * 7
        start = 1460713373
        nature = "una_bif-fa"
        test_string = 'test1235'

        s = ph.EvenlySignal(values=np.cumsum(np.random.rand(1, samples) - .5) * 100,
                            sampling_freq=freq,
                            signal_nature=nature,
                            start_time=start,
                            )

        # ineffective
        s.resample(freq_down)

        # length Y
        self.assertEqual(len(s), samples)
        self.assertEqual(len(s.get_values()), samples)
        # length X
        self.assertEqual(len(s.get_times()), samples)
        # sampling frequency in Hz
        self.assertEqual(s.get_sampling_freq(), freq)
        # duration in seconds
        self.assertEqual(s.get_duration(), samples / freq)
        # start timestamp
        self.assertEqual(s.get_start_time(), start)
        # end timestamp
        self.assertEqual(s.get_end_time(), start + s.get_duration())
        # meta data present
        self.assertEqual(s.get_metadata()['mode'], test_string)
        # start time
        self.assertEqual(s.get_signal_nature(), nature)

    def test_unevenly_signal_base(self):
        original_length = 1000
        samples = 200
        freq = 13
        start = 1460713373
        nature = "una_bif-fa"
        test_string = 'test1235'

        s = ph.UnevenlySignal(values=np.cumsum(np.random.rand(1, samples) - .5) * 100,
                              x_values=np.cumsum(np.random.rand(1, samples)) * 100,
                              sampling_freq=freq,
                              signal_nature=nature,
                              start_time=start,
                              )

        # length Y
        self.assertEqual(len(s), samples)
        self.assertEqual(len(s.get_values()), samples)
        # length X
        self.assertEqual(len(s.get_indices()), samples)
        # sampling frequency in Hz
        self.assertEqual(s.get_sampling_freq(), freq)
        # duration in seconds
        self.assertEqual(s.get_duration(), original_length / freq)
        # start timestamp
        self.assertEqual(s.get_start_time(), start)
        # end timestamp
        self.assertEqual(s.get_end_time(), start + s.get_duration())
        # meta data present
        self.assertEqual(s.get_metadata()['mode'], test_string)
        # start time
        self.assertEqual(s.get_signal_nature(), nature)
        # original length
        self.assertEqual(s.get_original_length(), original_length)

    def test_evenly_signal_resample(self):
        samples = 1000
        freq_down = 7
        freq = freq_down * 13
        freq_up = freq * 3
        freq_down_r = freq / 5
        start = 1460713373
        nature = "una_bif-fa"
        test_string = 'test1235'

        s = ph.EvenlySignal(values=np.cumsum(np.random.rand(1, samples) - .5) * 100,
                            sampling_freq=freq,
                            signal_nature=nature,
                            start_time=start,
                            )

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
        indexes = np.cumsum(np.round(np.random.rand(1, samples)+1))
        freq = 13
        original_length = (indexes[-1] + 1) * freq
        start = 1460713373
        nature = "una_bif-fa"
        test_string = 'test1235'

        s = ph.UnevenlySignal(values=np.cumsum(np.random.rand(1, samples) - .5) * 100,
                              x_values=indexes,
                              sampling_freq=freq,
                              signal_nature=nature,
                              start_time=start,
                              )

        # conversion
        s = s.to_evenly()  # TODO: add custom freq

        # length Y
        self.assertEqual(len(s), original_length)
        self.assertEqual(len(s.get_values()), len(s))
        # length X
        self.assertEqual(len(s.get_indices()), len(s))
        # sampling frequency in Hz
        self.assertEqual(s.get_sampling_freq(), freq)
        # duration in seconds
        self.assertEqual(s.get_duration(), original_length / freq)
        # start timestamp
        self.assertEqual(s.get_start_time(), start)
        # end timestamp
        self.assertEqual(s.get_end_time(), start + s.get_duration())
        # meta data present
        self.assertEqual(s.get_metadata()['mode'], test_string)
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
                self.assertGreaterEqual(y[j].get_begin(), 0)
                self.assertGreaterEqual(y[j].get_end(), 0)
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
                                                       x_values=np.array([10, 12, 13.5, 14.3, 15.6, 20.1123, 25, 36.8])
                                                       + start,
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
        test_string = 'test1235'

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

        s1 = s[0:x4]
        s2 = s[x1:x3]
        s3 = s[x2:-1]
        s4 = s[x3:samples]
        s5 = s[0:]
        s6 = s[:x2]
        s7 = s[:]

        self.assertEqual(len(s1), x4 - 0)
        self.assertEqual(len(s2), x3 - x1)
        self.assertEqual(len(s3), samples - x2 - 1)
        self.assertEqual(len(s4), samples - x3)
        self.assertEqual(len(s5), samples)
        self.assertEqual(len(s6), x2)
        self.assertEqual(len(s7), samples)
        self.assertEqual(s1.get_idx(0), 0)
        self.assertEqual(s2.get_idx(0), x1)
        self.assertEqual(s3.get_idx(0), x2)
        self.assertEqual(s4.get_idx(0), x3)
        self.assertEqual(s5.get_idx(0), 0)
        self.assertEqual(s6.get_idx(0), 0)
        self.assertEqual(s7.get_idx(0), 0)
        self.assertEqual(s1.get_idx(s1.get_end_time()), x4 - 1)
        self.assertEqual(s2.get_idx(s2.get_end_time()), x3 - 1)
        self.assertEqual(s3.get_idx(s3.get_end_time()), samples - 2)
        self.assertEqual(s4.get_idx(s4.get_end_time()), samples - 1)
        self.assertEqual(s5.get_idx(s5.get_end_time()), samples - 1)
        self.assertEqual(s6.get_idx(s6.get_end_time()), x2 - 1)
        self.assertEqual(s7.get_idx(s7.get_end_time()), samples - 1)

    def test_unevenly_slicing(self):
        samples = 1000
        x1 = 200
        x2 = 201
        x3 = 590
        x4 = 999
        freq = 7 * 13
        start = 1460713373
        nature = "una_bif-fa"

        x_vals = np.cumsum(np.random.rand(1, samples) * 9 + 1).astype(int)

        s = ph.UnevenlySignal(values=np.cumsum(np.random.rand(1, samples) - .5) * 100,
                              x_values=x_vals,
                              sampling_freq=freq,
                              signal_nature=nature,
                              start_time=start,
                              )

        s1 = s[0:x4]
        s2 = s[x1:x3]
        s3 = s[x2:-1]
        s4 = s[x3:samples]
        s5 = s[0:]
        s6 = s[:x2]
        s7 = s[:]

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

        self.assertEqual(s1.get_indices()[0], s.get_indices()[0])
        self.assertEqual(s2.get_indices()[0], s.get_indices()[x1])
        self.assertEqual(s3.get_indices()[0], s.get_indices()[x2])
        self.assertEqual(s4.get_indices()[0], s.get_indices()[x3])
        self.assertEqual(s5.get_indices()[0], s.get_indices()[0])
        self.assertEqual(s6.get_indices()[0], s.get_indices()[0])
        self.assertEqual(s7.get_indices()[0], s.get_indices()[0])
        self.assertEqual(s1.get_indices()[-1], s.get_indices()[x4 - 1])
        self.assertEqual(s2.get_indices()[-1], s.get_indices()[x3 - 1])
        self.assertEqual(s3.get_indices()[-1], s.get_indices()[samples - 2])
        self.assertEqual(s4.get_indices()[-1], s.get_indices()[samples - 1])
        self.assertEqual(s5.get_indices()[-1], s.get_indices()[samples - 1])
        self.assertEqual(s6.get_indices()[-1], s.get_indices()[x2 - 1])
        self.assertEqual(s7.get_indices()[-1], s.get_indices()[samples - 1])

    def test_signal_plot(self):
        s = np.genfromtxt("../sample_data/medical.txt", delimiter="\t", max_rows=10000)

        e = ph.EvenlySignal(values=s[:, 0], sampling_freq=1024, signal_nature="ecg")
        e, ignored, ignored, ignored = ph.PeakDetection(delta=1)(e)
        e = ph.UnevenlySignal(values=e, x_values=e, x_type='indices', sampling_freq=1024, signal_nature="ibi")
        e.plot("|b")
        e = ph.EvenlySignal(values=s[:, 1], sampling_freq=1024, signal_nature="gsr")
        e.plot()
        e = ph.EvenlySignal(values=s[:, 2], sampling_freq=1024, signal_nature="bvp")
        e.plot()
        e = ph.EvenlySignal(values=s[:, 3], sampling_freq=1024, signal_nature="resp")
        e.plot()

        import matplotlib.pyplot as plt
        plt.show()


if __name__ == '__main__':
    unittest.main()
