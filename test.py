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
    print "RRMean: ", In.RRMean(RRseries).value
    print "RRSTD: ", In.RRSTD(RRseries).value
    print "PNNx: ", In.PNNx(50, RRseries).value
    print "NNx: ", In.NNx(50, RRseries).value
    vlf,lf,hf, lfhf, vlf_peak = In.VLF(RRseries), In.LF(RRseries), In.HF(RRseries), In.LFHF(RRseries), In.VLFPeak(RRseries)
    print vlf,lf,hf, lfhf, vlf_peak
    delay()

if __name__ == '__main__':
    delay(False)

    RRseries = load_rr_data_series("A05")

    print 'Starting analysis'
    test(RRseries)

