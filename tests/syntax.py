# coding=utf-8

import numpy as np
import pyphysio.pyPhysio as ph

__author__ = 'AleB'


s = ph.EvenlySignal((np.sin(np.random.rand(100, 1)*3.14-(3.14/2))+1)*93, 15)
print "Mean:", ph.Mean()(s)
print "NN100:", ph.NNx(threshold=100)(s)
print "LF:", ph.PowerInBand(interp_freq=20, freq_max=4, freq_min=0.001)(s)
print "HF:", ph.PowerInBand(interp_freq=20, freq_min=4, freq_max=15)(s)
print "LFHF:", ph.LFHF(interp_freq=20, freq_mid=4, freq_min=0.001, freq_max=15)(s)
ph.LengthSegments(1, 2)(s)
