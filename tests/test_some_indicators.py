# coding=utf-8
from __future__ import division

from context import ph
import numpy as np
import unittest

__author__ = 'aleb'


# TODO: not really useful... we should write a test for all indicators --> utest_indicators_hrv & utest_indicators_eda

# noinspection PyArgumentEqualDefault
class SomeIndicatorsTest(unittest.TestCase):
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
        
        self.assertGreaterEqual(ph.RMSSD()(s), 0)
        self.assertGreaterEqual(ph.StDev()(s), 0)
        self.assertGreaterEqual(ph.Triang()(s), 0)


if __name__ == '__main__':
    unittest.main()