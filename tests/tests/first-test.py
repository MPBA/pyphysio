# coding=utf-8
from __future__ import division

import pyphysio.pyPhysio as ph
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

        s = ph.EvenlySignal(y_values=np.cumsum(np.random.rand(1, samples) - .5) * 100,
                            sampling_freq=freq,
                            signal_nature=nature,
                            start_time=start,
                            meta={'mode': test_string, 'subject': 'Baptist;Alessandro'}
                            )

        # ineffective
        s.resample(freq_down)

        # length Y
        self.assertEqual(len(s), samples)
        self.assertEqual(len(s.get_y_values()), samples)
        # length X
        self.assertEqual(len(s.get_x_values()), samples)
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

        s = ph.UnevenlySignal(y_values=np.cumsum(np.random.rand(1, samples) - .5) * 100,
                              indexes=np.cumsum(np.random.rand(1, samples)) * 100,
                              sampling_freq=freq,
                              original_length=original_length,
                              signal_nature=nature,
                              start_time=start,
                              meta={'mode': test_string, 'subject': 'Baptist;Alessandro'},
                              check=True
                              )

        # length Y
        self.assertEqual(len(s), samples)
        self.assertEqual(len(s.get_y_values()), samples)
        # length X
        self.assertEqual(len(s.get_x_values()), samples)
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

    def test_sparse_signal_base(self):
        samples = 1000
        freq = 13
        start = 1460713373
        nature = "una_bif-fa"
        test_string = 'test1235'
        times = np.cumsum(np.random.rand(1, samples))
        duration = times[-1]

        s = ph.SparseSignal(y_values=np.cumsum(np.random.rand(1, samples) - .5) * 100,
                            x_values=times,
                            sampling_freq=freq,
                            signal_nature=nature,
                            start_time=start,
                            meta={'mode': test_string, 'subject': 'Baptist;Alessandro'},
                            check=True
                            )

        # length Y
        self.assertEqual(len(s), samples)
        self.assertEqual(len(s.get_y_values()), samples)
        # length X
        self.assertEqual(len(s.get_x_values()), samples)
        # sampling frequency in Hz
        self.assertEqual(s.get_sampling_freq(), freq)
        # duration in seconds
        self.assertEqual(s.get_duration(), duration)
        # start timestamp
        self.assertEqual(s.get_start_time(), start)
        # end timestamp
        self.assertEqual(s.get_end_time(), start + s.get_duration())
        # meta data present
        self.assertEqual(s.get_metadata()['mode'], test_string)
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
        test_string = 'test1235'

        s = ph.EvenlySignal(y_values=np.cumsum(np.random.rand(1, samples) - .5) * 100,
                            sampling_freq=freq,
                            signal_nature=nature,
                            start_time=start,
                            meta={'mode': test_string, 'subject': 'Baptist;Alessandro'}
                            )

        def check_resampled(freq, freq_down, nature, sig, samples, start, test_string):
            # resample
            resampled = sig.resample(freq_down)
            # new number of samples
            new_samples = _ceil(samples * freq_down / freq)

            # length Y
            self.assertEqual(len(resampled), new_samples)
            # length X
            self.assertEqual(len(resampled.get_x_values()), len(resampled))
            # sampling frequency in Hz
            self.assertEqual(resampled.get_sampling_freq(), freq_down)
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
        check_resampled(freq, freq_down, nature, s, samples, start, test_string)
        # over-sampling int
        check_resampled(freq, freq_up, nature, s, samples, start, test_string)
        # down-sampling rationale
        check_resampled(freq, freq_down_r, nature, s, samples, start, test_string)

    def test_unevenly_signal_advanced(self):
        original_length = 1000
        samples = 200
        freq = 13
        start = 1460713373
        nature = "una_bif-fa"
        test_string = 'test1235'

        s = ph.UnevenlySignal(y_values=np.cumsum(np.random.rand(1, samples) - .5) * 100,
                              indexes=np.cumsum(np.random.rand(1, samples)) * 100,
                              sampling_freq=freq,
                              original_length=original_length,
                              signal_nature=nature,
                              start_time=start,
                              meta={'mode': test_string, 'subject': 'Baptist;Alessandro'},
                              check=True
                              )
        # length Y
        self.assertEqual(len(s), samples)
        self.assertEqual(len(s.get_y_values()), samples)
        # length X
        self.assertEqual(len(s.get_x_values()), samples)
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

    def test_sparse_signal_advanced(self):
        samples = 1000
        freq = 13
        start = 1460713373
        nature = "una_bif-fa"
        test_string = 'test1235'
        times = np.cumsum(np.random.rand(1, samples))
        duration = times[-1]

        s = ph.SparseSignal(y_values=np.cumsum(np.random.rand(1, samples) - .5) * 100,
                            x_values=times,
                            sampling_freq=freq,
                            signal_nature=nature,
                            start_time=start,
                            meta={'mode': test_string, 'subject': 'Baptist;Alessandro'},
                            check=True
                            )
        # length Y
        self.assertEqual(len(s), samples)
        self.assertEqual(len(s.get_y_values()), samples)
        # length X
        self.assertEqual(len(s.get_x_values()), samples)
        # sampling frequency in Hz
        self.assertEqual(s.get_sampling_freq(), freq)
        # duration in seconds
        self.assertEqual(s.get_duration(), duration)
        # start timestamp
        self.assertEqual(s.get_start_time(), start)
        # end timestamp
        self.assertEqual(s.get_end_time(), start + s.get_duration())
        # meta data present
        self.assertEqual(s.get_metadata()['mode'], test_string)
        # start time
        self.assertEqual(s.get_signal_nature(), nature)


if __name__ == '__main__':
    unittest.main()