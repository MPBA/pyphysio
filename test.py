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
    rrm, rrs, pnnx, nnx = In.RRMean(RRseries), In.RRSTD(RRseries), In.PNNx(50, RRseries), In.NNx(50, RRseries)
    #vlf,lf,hf, lfhf, vlf_peak = In.VLF(RRseries), In.LF(RRseries), In.HF(RRseries), In.LFHF(RRseries), In.VLFPeak(RRseries)
    print "RRMean: ", rrm.value
    print "RRSTD: ", rrs.value
    print "PNNx: ", pnnx.value
    print "NNx: ", nnx.value
    #print vlf,lf,hf, lfhf, vlf_peak
    delay()

if __name__ == '__main__':
    delay(False)

    RRseries = load_rr_data_series("A05.txt")

    print 'Starting analysis'
    test(RRseries)

