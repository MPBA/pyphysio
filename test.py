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
    print "RRMedian: ", In.RRMedian(RRseries).value
    print "PNNx: ", In.PNNx(50, RRseries).value
    print "NNx: ", In.NNx(50, RRseries).value
    print "RMSSD: ", In.RMSSD(RRseries).value
    print "SDSD: ", In.SDSD(RRseries).value
    delay()
    print "Cached:"
    print "HRMean: ", In.HRMean(RRseries).value
    print "HRSTD: ", In.HRSTD(RRseries).value
    print "HRMedian: ", In.HRMedian(RRseries).value
    delay()
    print '2. FD'
    print "VLF: ", In.VLF(RRseries).value
    print "LF: ", In.LF(RRseries).value
    print "HF: ", In.HF(RRseries).value
    print "VLFPeak: ", In.VLFPeak(RRseries).value
    delay()
    print "Cached:"
    print "LFHF: ", In.LFHF(RRseries).value
    delay()

if __name__ == '__main__':
    delay(False)
    print 'Loading file'
    RRseries = load_rr_data_series("A05")
    delay()

    print 'Starting analysis'
    test(RRseries)

    print '\n\nCache Test'
    test(RRseries)

