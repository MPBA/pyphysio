import numpy as np
import time
from pyhrv import *
from utility import *

_debug_time = 0


def delay(v=True):
    global _debug_time
    d = time.time() - _debug_time
    _debug_time = time.time()
    if v:
        print 'Delay: ', int(d * 100000) / 100.0, 'ms'


def test():
    print '1. TD'
    TD_ind, TD_lab = RRAnalysis.TD_indexes(RRseries)
    print TD_lab
    print TD_ind
    delay()
    print '2. POIN'
    POIN_ind, POIN_lab = RRAnalysis.POIN_indexes(RRseries)
    print POIN_lab
    print POIN_ind
    delay()
    print '3. FD'
    FD_ind, FD_lab = RRAnalysis.FD_indexes(RRseries, 1)
    print FD_lab
    print FD_ind
    delay()

delay(False)

np.random.seed()
t = np.arange(0, 10*np.pi, 0.01*np.pi)
RRraw = np.random.uniform(500, 1500, 1000)+100*np.sin(t)

RRseries = DataSeries(RRraw)

print 'Starting analysis'
test()

print 'Ok, lez test the cache'
test()
