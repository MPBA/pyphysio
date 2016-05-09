# coding=utf-8
from __future__ import division

try:
    import pyphysio.pyPhysio as ph
except ImportError:
    import pyPhysio as ph
import numpy as np
from math import ceil as _ceil

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

    # def test_sparse_signal_base(self):
    #     samples = 1000
    #     freq = 13
    #     start = 1460713373
    #     nature = "una_bif-fa"
    #     test_string = 'test1235'
    #     times = np.cumsum(np.random.rand(1, samples))
    #     duration = times[-1]
    #
    #     s = ph.SparseSignal(y_values=np.cumsum(np.random.rand(1, samples) - .5) * 100,
    #                         times=times,
    #                         sampling_freq=freq,
    #                         signal_nature=nature,
    #                         start_time=start,
    #                         meta={'mode': test_string, 'subject': 'Baptist;Alessandro'},
    #                         check=True
    #                         )
    #
    #     # length Y
    #     self.assertEqual(len(s), samples)
    #     self.assertEqual(len(s.get_values()), samples)
    #     # length X
    #     self.assertEqual(len(s.get_indices()), samples)
    #     # sampling frequency in Hz
    #     self.assertEqual(s.get_sampling_freq(), freq)
    #     # duration in seconds
    #     self.assertEqual(s.get_duration(), duration)
    #     # start timestamp
    #     self.assertEqual(s.get_start_time(), start)
    #     # end timestamp
    #     self.assertEqual(s.get_end_time(), start + s.get_duration())
    #     # meta data present
    #     self.assertEqual(s.get_metadata()['mode'], test_string)
    #     # start time
    #     self.assertEqual(s.get_signal_nature(), nature)

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

    # def test_sparse_signal_advanced(self):
    #     samples = 1000
    #     freq = 13
    #     start = 1460713373
    #     nature = "una_bif-fa"
    #     test_string = 'test1235'
    #     times = np.cumsum(np.random.rand(1, samples))
    #     duration = times[-1]
    #
    #     s = ph.SparseSignal(y_values=np.cumsum(np.random.rand(1, samples) - .5) * 100,
    #                         times=times,
    #                         sampling_freq=freq,
    #                         signal_nature=nature,
    #                         start_time=start,
    #                         meta={'mode': test_string, 'subject': 'Baptist;Alessandro'},
    #                         check=True
    #                         )
    #     # length Y
    #     self.assertEqual(len(s), samples)
    #     self.assertEqual(len(s.get_values()), samples)
    #     # length X
    #     self.assertEqual(len(s.get_indices()), samples)
    #     # sampling frequency in Hz
    #     self.assertEqual(s.get_sampling_freq(), freq)
    #     # duration in seconds
    #     self.assertEqual(s.get_duration(), duration)
    #     # start timestamp
    #     self.assertEqual(s.get_start_time(), start)
    #     # end timestamp
    #     self.assertEqual(s.get_end_time(), start + s.get_duration())
    #     # meta data present
    #     self.assertEqual(s.get_metadata()['mode'], test_string)
    #     # start time
    #     self.assertEqual(s.get_signal_nature(), nature)

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

if __name__ == '__main__':
    unittest.main()
