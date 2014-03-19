import time
import TDIndexes as TDIn
import FDIndexes as FDIn
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
    print "RRMean: ", TDIn.RRMean(RRseries).value
    print "RRSTD: ", TDIn.RRSTD(RRseries).value
    print "RRMedian: ", TDIn.RRMedian(RRseries).value
    print "PNNx: ", TDIn.PNNx(50, RRseries).value
    print "NNx: ", TDIn.NNx(50, RRseries).value
    print "RMSSD: ", TDIn.RMSSD(RRseries).value
    print "SDSD: ", TDIn.SDSD(RRseries).value
    t1 = delay()
    print "*Cached:"
    print "HRMean: ", TDIn.HRMean(RRseries).value
    print "HRSTD: ", TDIn.HRSTD(RRseries).value
    print "HRMedian: ", TDIn.HRMedian(RRseries).value
    t2 = delay()
    print '2. FD'
    print "VLF: ", FDIn.VLF(RRseries).value
    print "LF: ", FDIn.LF(RRseries).value
    print "HF: ", FDIn.HF(RRseries).value
    print "Total: ", FDIn.Total(RRseries).value
    print "VLFPeak: ", FDIn.VLFPeak(RRseries).value
    print "LFPeak: ", FDIn.LFPeak(RRseries).value
    print "HFPeak: ", FDIn.HFPeak(RRseries).value
    print "VLFNormal: ", FDIn.VLFNormal(RRseries).value
    print "LFNormal: ", FDIn.LFNormal(RRseries).value
    print "HFNormal: ", FDIn.HFNormal(RRseries).value
    t3 = delay()
    print "*Cached:"
    print "LFHF: ", FDIn.LFHF(RRseries).value
    print "NormalLF: ", FDIn.NormalLF(RRseries).value
    print "NormalHF: ", FDIn.NormalHF(RRseries).value
    t4 = delay()
    print "\t\tTotal time: ", t1 + t2 + t3 + t4, "ms"

if __name__ == '__main__':
    delay(False)
    print 'Loading file'
    RRseries = load_rr_data_series("A05")
    delay()

    print 'Starting analysis'
    test(RRseries)

    print '\n\nCache Test'
    test(RRseries)

