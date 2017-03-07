# coding=utf-8
from __future__ import division

from context import ph
from math import ceil as _ceil
import numpy as np

import unittest

__author__ = 'aleb'


# noinspection PyArgumentEqualDefault
class GeneralTest(unittest.TestCase):
    def test_issue24(self):
        FSAMP = 100
        n = np.arange(1000)
        t = n / FSAMP
        freq = 1

        # create reference signal
        sinusoid = ph.EvenlySignal(np.sin(2 * np.pi * freq * t), sampling_freq=FSAMP, signal_nature='', start_time=0)

        selection = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

        # remove some samples
        sinusoid_unevenly = ph.UnevenlySignal(
            np.delete(sinusoid.get_values(), selection),
            sampling_freq=FSAMP, signal_nature='', start_time=0,
            x_values=np.delete(sinusoid.get_times(), selection),
            x_type='instants')

        unev_t = sinusoid_unevenly.get_times()
        even_t = np.delete(sinusoid.get_times(), selection)
        error_sum = np.sum(sinusoid_unevenly.get_times() - np.delete(sinusoid.get_times(), selection))
        error_sum1 = np.sum(unev_t - even_t)
        where_error = np.where(sinusoid_unevenly.get_times() != np.delete(sinusoid.get_times(), selection))
        where_in_even_t = even_t[where_error]
        where_in_unev_t = unev_t[where_error]
        errors = where_in_unev_t - where_in_even_t

        # 69 errors of -0.01

        self.assertEquals(error_sum, error_sum1)
        self.assertEquals(error_sum, 0)
        self.assertEquals(error_sum, np.sum(errors))
