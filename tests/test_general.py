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
                            meta={'mode': test_string, 'subject': 'Baptist;Alessandro'}
                            )

        # ineffective
        s.resample(freq_down)

        # length Y
        self.assertEqual(len(s), samples)
        self.assertEqual(len(s.get_values()), samples)
        # length X
        self.assertEqual(len(s.get_indices()), samples)
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
                              indices=np.cumsum(np.random.rand(1, samples)) * 100,
                              orig_sampling_freq=freq,
                              orig_length=original_length,
                              signal_nature=nature,
                              start_time=start,
                              meta={'mode': test_string, 'subject': 'Baptist;Alessandro'},
                              check=True
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
                            meta={'mode': test_string, 'subject': 'Baptist;Alessandro'}
                            )

        def check_resampled(freq_new):
            # resample
            resampled = s.resample(freq_new)
            # new number of samples
            new_samples = _ceil(samples * freq_new / freq)

            # length Y
            self.assertEqual(len(resampled), new_samples)
            # length X
            self.assertEqual(len(resampled.get_indices()), len(resampled))
            # sampling frequency in Hz
            self.assertEqual(resampled.get_sampling_freq(), freq_new)
            # duration differs less than 1s from the original
            self.assertLess(abs(resampled.get_duration() - samples / freq), 1)
            # start timestamp
            self.assertEqual(resampled.get_start_time(), start)
            # end timestamp
            self.assertEqual(resampled.get_end_time(), start + resampled.get_duration())
            # meta data present
            self.assertEqual(resampled.get_metadata()['mode'], test_string)
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
        indexes = np.round(np.cumsum(np.random.rand(1, samples)))
        freq = 13
        original_length = (indexes[-1] + 1) * freq
        start = 1460713373
        nature = "una_bif-fa"
        test_string = 'test1235'

        s = ph.UnevenlySignal(values=np.cumsum(np.random.rand(1, samples) - .5) * 100,
                              indices=indexes,
                              orig_sampling_freq=freq,
                              orig_length=original_length,
                              signal_nature=nature,
                              start_time=start,
                              meta={'mode': test_string, 'subject': 'Baptist;Alessandro'},
                              check=True
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
        start = 1460713373
        nature = "una_bif-fa"
        test_string = 'test1235'

        s = ph.EvenlySignal(values=np.cumsum(np.random.rand(1, samples) - .5) * 100,
                            sampling_freq=freq,
                            signal_nature=nature,
                            start_time=start,
                            meta={'mode': test_string, 'subject': 'Baptist;Alessandro'}
                            )

        def function(y):
            for j in xrange(len(y)):
                self.assertLessEqual(y[j].get_begin(), y[j].get_end())
                self.assertGreaterEqual(y[j].get_begin(), 0)
                self.assertGreaterEqual(y[j].get_end(), 0)
                # list[x:y] where y > len(list) is a valid usage in python so next lines are cut
                # self.assertLessEqual(y[j].get_begin(), len(s))
                # self.assertLessEqual(y[j].get_end(), len(s))

        w1 = ph.TimeSegments(step=2, width=3)(s)
        y1 = [x for x in w1]
        function(y1)

        w2 = ph.LengthSegments(step=100, width=121)(s)
        y2 = [x for x in w2]
        function(y2)

        w3 = ph.FromEventsSegments(events=ph.EventsSignal(['a', 'a', 'b', 'a', 'r', 's', 'r', 'b'], [10, 12, 13.5, 14.3, 15.6, 20.1123, 25, 36.8]))(s)
        y3 = [x for x in w3]
        function(y3)

        # iter from SegmentsGenerator
        w4 = ph.ExistingSegments(segments=w3)(s)
        y4 = [x for x in w4]
        function(y4)

        # iter from list
        w5 = ph.ExistingSegments(segments=y3)(s)
        y5 = [x for x in w5]
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
        for i in xrange(n):
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
                            meta={'mode': test_string, 'subject': 'Baptist;Alessandro'}
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
        test_string = 'test1235'

        s = ph.EvenlySignal(values=np.cumsum(np.random.rand(1, samples) - .5) * 100,
                            sampling_freq=freq,
                            signal_nature=nature,
                            start_time=start,
                            meta={'mode': test_string, 'subject': 'Baptist;Alessandro'}
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
        self.assertEqual(s1.get_indices(0), 0)
        self.assertEqual(s2.get_indices(0), x1)
        self.assertEqual(s3.get_indices(0), x2)
        self.assertEqual(s4.get_indices(0), x3)
        self.assertEqual(s5.get_indices(0), 0)
        self.assertEqual(s6.get_indices(0), 0)
        self.assertEqual(s7.get_indices(0), 0)
        self.assertEqual(s1.get_indices(-1), x4 - 1)
        self.assertEqual(s2.get_indices(-1), x3 - 1)
        self.assertEqual(s3.get_indices(-1), samples - 2)
        self.assertEqual(s4.get_indices(-1), samples - 1)
        self.assertEqual(s5.get_indices(-1), samples - 1)
        self.assertEqual(s6.get_indices(-1), x2 - 1)
        self.assertEqual(s7.get_indices(-1), samples - 1)

    def test_unevenly_slicing(self):
        samples = 1000
        x1 = 200
        x2 = 201
        x3 = 590
        x4 = 999
        freq = 7 * 13
        start = 1460713373
        nature = "una_bif-fa"
        test_string = 'test1235'

        x_vals = np.cumsum(np.random.rand(1, samples) * 10).astype(int)
        orig_len = x_vals[-1] + int(np.random.rand() * 10)

        print "orig_len:", orig_len

        s = ph.UnevenlySignal(values=np.cumsum(np.random.rand(1, samples) - .5) * 100,
                              indices=x_vals,
                              orig_sampling_freq=freq,
                              orig_length=orig_len,
                              signal_nature=nature,
                              start_time=start,
                              meta={'mode': test_string, 'subject': 'Baptist;Alessandro'}
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

        self.assertEqual(s1.get_indices(0), s.get_indices(0))
        self.assertEqual(s2.get_indices(0), s.get_indices(x1))
        self.assertEqual(s3.get_indices(0), s.get_indices(x2))
        self.assertEqual(s4.get_indices(0), s.get_indices(x3))
        self.assertEqual(s5.get_indices(0), s.get_indices(0))
        self.assertEqual(s6.get_indices(0), s.get_indices(0))
        self.assertEqual(s7.get_indices(0), s.get_indices(0))
        self.assertEqual(s1.get_indices(-1), s.get_indices(x4 - 1))
        self.assertEqual(s2.get_indices(-1), s.get_indices(x3 - 1))
        self.assertEqual(s3.get_indices(-1), s.get_indices(samples - 2))
        self.assertEqual(s4.get_indices(-1), s.get_indices(samples - 1))
        self.assertEqual(s5.get_indices(-1), s.get_indices(samples - 1))
        self.assertEqual(s6.get_indices(-1), s.get_indices(x2 - 1))
        self.assertEqual(s7.get_indices(-1), s.get_indices(samples - 1))


if __name__ == '__main__':
    unittest.main()
