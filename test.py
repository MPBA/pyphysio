import time
import Indexes as In
from Files import *

_debug_time = 0


def delay(v=True):
    global _debug_time
    d = time.time() - _debug_time
    _debug_time = time.time()
    if v:
        print 'Delay: ', int(d * 100000) / 100.0, 'ms'


def test(RRseries):
    print '1. TD'
    rrm, rrs, pnnx, nnx = In.RRMean(RRseries), In.RRSTD(RRseries), In.pNNx(50, RRseries), In.NNx(50, RRseries)
    print "RRMean: ", rrm.value
    print "RRSTD: ", rrs.value
    print "pNNx: ", pnnx.value
    print "NNx: ", nnx.value
    delay()

if __name__ == '__main__':
    delay(False)

    RRseries = load_rr_data_series("/media/ale/44A0-BCA5/gx/Subject_data/B01.txt")

    print 'Starting analysis'
    test(RRseries)

