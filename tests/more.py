# coding=utf-8
__author__ = 'AleB'

import numpy as np
try:
    import pyphysio.pyPhysio as ph
except ImportError:
    import pyPhysio as ph

s = ph.EvenlySignal(np.cumsum(np.random.rand(1, 1000) - .5) * 100, 10)

w1 = ph.TimeSegments(step=2, width=3)(s)
w2 = ph.LengthSegments(step=100, width=121)(s)
w3 = ph.FromEventsSegments(events=ph.UnevenlySignal(
    ['a', 'a', 'b', 'a', 'r', 's', 'r', 'b'], [10, 12, 13.5, 14.3, 15.6, 20.1123, 25, 36.8], 10, 40))(s)
w4 = ph.ExistingSegments(segments=w3)(s)

y1 = [x for x in w1]
y2 = [x for x in w2]
y3 = [x for x in w3]
ph.fmap(w1, [ph.Mean(), ph.StDev(), ph.NNx(threshold=31)])
ph.fmap(w2, [ph.Mean(), ph.StDev(), ph.NNx(threshold=31)])
ph.fmap(w3, [ph.Mean(), ph.StDev(), ph.NNx(threshold=31)])
ph.fmap(y1, [ph.Mean(), ph.StDev(), ph.NNx(threshold=31)])
ph.fmap(y2, [ph.Mean(), ph.StDev(), ph.NNx(threshold=31)])
ph.fmap(y3, [ph.Mean(), ph.StDev(), ph.NNx(threshold=31)])

# noinspection PyArgumentEqualDefault
sd2 = s.resample(1, 'linear')
sd3 = s.resample(1, 'nearest')
sd4 = s.resample(1, 'zero')
sd5 = s.resample(1, 'slinear')
sd6 = s.resample(1, 'quadratic')
sd7 = s.resample(1, 'cubic')

# noinspection PyArgumentEqualDefault
so2 = s.resample(20, 'linear')
so3 = s.resample(20, 'nearest')
so4 = s.resample(20, 'zero')
so5 = s.resample(20, 'slinear')
so6 = s.resample(20, 'quadratic')
so7 = s.resample(20, 'cubic')

so1 = s.resample(21)
