import time
import Indexes as In
from Files import *

_debug_time = 0


def delay(v=True):
    global _debug_time
    d = time.time() - _debug_time
    _debug_time = time.time()
    t = int(d * 100000) / 100.0
    if v:
        print '\t\tDelay: ', t, 'ms'
    return t


def test(RRseries):
    print '1. TD'
    print "RRMean: ", In.RRMean(RRseries).value
    print "RRSTD: ", In.RRSTD(RRseries).value
    print "RRMedian: ", In.RRMedian(RRseries).value
    print "PNNx: ", In.PNNx(50, RRseries).value
    print "NNx: ", In.NNx(50, RRseries).value
    print "RMSSD: ", In.RMSSD(RRseries).value
    print "SDSD: ", In.SDSD(RRseries).value
    t1=delay()
    print "*Cached:"
    print "HRMean: ", In.HRMean(RRseries).value
    print "HRSTD: ", In.HRSTD(RRseries).value
    print "HRMedian: ", In.HRMedian(RRseries).value
    t2=delay()
    print '2. FD'
    print "VLF: ", In.VLF(RRseries).value
    print "LF: ", In.LF(RRseries).value
    print "HF: ", In.HF(RRseries).value
    print "Total: ", In.Total(RRseries).value
    print "VLFPeak: ", In.VLFPeak(RRseries).value
    print "LFPeak: ", In.LFPeak(RRseries).value
    print "HFPeak: ", In.HFPeak(RRseries).value
    print "VLFNormal: ", In.VLFNormal(RRseries).value
    print "LFNormal: ", In.LFNormal(RRseries).value
    print "HFNormal: ", In.HFNormal(RRseries).value
    t3 = delay()
    print "*Cached:"
    print "LFHF: ", In.LFHF(RRseries).value
    print "NormalLF: ", In.NormalLF(RRseries).value
    print "NormalHF: ", In.NormalHF(RRseries).value
    t4 = delay()
    print "\t\tTotal time: ", t1+t2+t3+t4, "ms"

if __name__ == '__main__':
    delay(False)
    print 'Loading file'
    RRseries = load_rr_data_series("A05")
    delay()

    print 'Starting analysis'
    test(RRseries)

    print '\n\nCache Test'
    test(RRseries)

