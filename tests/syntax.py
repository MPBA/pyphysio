# coding=utf-8

import numpy as np
try:
    import pyphysio.pyPhysio as ph
except ImportError:
    import pyPhysio as ph
import unittest

__author__ = 'AleB'


class SyntaxTest(unittest.TestCase):
    def test_fmap_and_algo(self):
        s = ph.EvenlySignal((np.sin(np.random.rand(100, 1) * 3.14 - (3.14 / 2)) + 1) * 93, 15)
        lf = ph.PowerInBand(freq_max=1, freq_min=0.001)
        hf = ph.PowerInBand(freq_max=4, freq_min=1)

        algos = [
            ph.Mean(),
            ph.StDev(),
            ph.NNx(threshold=100),
            ph.PowerInBand(interp_freq=20, freq_max=4, freq_min=0.001),
            ph.PowerInBand(interp_freq=20, freq_min=4, freq_max=15),
            ph.algo(lambda d, p: lf(d) / hf(d))(),
            ph.algo(lambda d, p: len(d))()
        ]

        g = ph.TimeSegments(step=1, width=1.5)
        l = len([x for x in g(s)])
        r = ph.fmap(g(s), algos)
        n = 3 + len(algos)
        self.assertTrue(reduce(lambda x, y: x and len(y) == n, r, True))
        self.assertEquals(len(r), l)

        ph.fmap(ph.LengthSegments(step=1, width=1.5)(s), algos)
