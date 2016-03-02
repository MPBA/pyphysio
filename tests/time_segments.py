# coding=utf-8
import pyphysio.pyPhysio as ph
import numpy as np

__author__ = 'AleB'

s = ph.UnevenlySignal((np.sin(np.random.rand(200, 1)*3.14-(3.14/2))+1)*93,
                      np.cumsum(np.random.rand(200, 1)))
t = ph.UnevenlySignal((np.sin(np.random.rand(200, 1)*3.14-(3.14/2))+1)*93,
                      np.cumsum(np.random.rand(200, 1)))
w1 = ph.TimeSegments(step=1)
print [x for x in w1]
print [x for x in w1(s)]
print [x for x in w1(t)]
print [x for x in ph.TimeSegments(step=10)(t)]

w2 = ph.ExistingSegments(segments=map(lambda y: ph.Segment(10*y, 10*y+10, "Hi" if y % 2 else "Lo"), xrange(100)))
print [x for x in w2]
print [x for x in w2(s)]

w3 = ph.FromEventsSegments(events=ph.EventsSignal(["Hi" if x % 2 else "Lo" for x in xrange(400)],
                                                  np.cumsum(np.random.rand(400, 1)*8)))
print [x for x in w3]
print [x for x in w3(s)]