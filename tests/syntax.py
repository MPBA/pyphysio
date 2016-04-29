# coding=utf-8

import numpy as np
import pyphysio.pyPhysio as ph
import unittest

__author__ = 'AleB'


class SyntaxTest(unittest.TestCase):
    def test_fmap(self):
        s = ph.EvenlySignal((np.sin(np.random.rand(100, 1) * 3.14 - (3.14 / 2)) + 1) * 93, 15)
        algos = [
            ph.Mean(),
            ph.StDev(),
            ph.NNx(threshold=100),
            # ph.PowerInBand(interp_freq=20, freq_max=4, freq_min=0.001),
            # ph.PowerInBand(interp_freq=20, freq_min=4, freq_max=15),
            # ph.algo(lambda s, p:
            #         ph.PowerInBand(freq_max=1, freq_min=0.001)(s) / ph.PowerInBand(freq_max=4, freq_min=1)(s))
        ]

        ph.fmap(ph.LengthSegments(step=1, width=1.5)(s), algos)