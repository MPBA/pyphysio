# coding=utf-8

from . import ph
import numpy as np
import unittest

__author__ = 'AleB'


# TODO: not really useful... we should write a test for all indicatots
class SyntaxTest(unittest.TestCase):
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
