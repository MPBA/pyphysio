# coding=utf-8
import pyphysio.pyPhysio as ph
import numpy as np

__author__ = 'AleB'

s = ph.UnevenlySignal((np.sin(np.random.rand(200, 1)*3.14-(3.14/2))+1)*93,
                      np.cumsum(np.random.rand(200, 1)))
print [x for x in ph.TimeSegments(step=1)(s)]
