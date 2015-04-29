import time

from pyPhysio.Files import *
from pyPhysio.indexes import TDFeatures as TDIn
from pyPhysio.indexes import FDFeatures as FDIn


_debug_time = 0


def delay(v=True):
    global _debug_time
    d = time.time() - _debug_time
    _debug_time = time.time()
    t = int(d * 100000) / 100.0
    if v:
        print '\t\tDelay: ', t, 'ms'
    return t


def test(rr_series):
    print '1. TD'
    print "Mean: ", TDIn.Mean(rr_series).value
    print "SD: ", TDIn.SD(rr_series).value
    print "Median: ", TDIn.Median(rr_series).value
    print "PNN50: ", TDIn.PNN50(rr_series).value
    print "NN50: ", TDIn.NN50(rr_series).value
    print "RMSSD: ", TDIn.RMSSD(rr_series).value
    print "SDSD: ", TDIn.SDSD(rr_series).value
    t1 = delay()
    print "*Cached:"
    print "HRMean: ", TDIn.HRMean(rr_series).value
    print "HRSD: ", TDIn.HRSD(rr_series).value
    print "HRMedian: ", TDIn.HRMedian(rr_series).value
    t2 = delay()
    print '2. FD'
    print "VLF: ", FDIn.VLF(rr_series).value
    print "LF: ", FDIn.LF(rr_series).value
    print "HF: ", FDIn.HF(rr_series).value
    print "Total: ", FDIn.Total(rr_series).value
    print "VLFPeak: ", FDIn.VLFPeak(rr_series).value
    print "LFPeak: ", FDIn.LFPeak(rr_series).value
    print "HFPeak: ", FDIn.HFPeak(rr_series).value
    print "VLFNormal: ", FDIn.VLFNormal(rr_series).value
    print "LFNormal: ", FDIn.LFNormal(rr_series).value
    print "HFNormal: ", FDIn.HFNormal(rr_series).value
    t3 = delay()
    print "*Cached:"
    print "LFHF: ", FDIn.LFHF(rr_series).value
    print "NormalizedLF: ", FDIn.NormalizedLF(rr_series).value
    print "NormalizedHF: ", FDIn.NormalizedHF(rr_series).value
    t4 = delay()
    print "\t\tTotal time: ", t1 + t2 + t3 + t4, "ms"


if __name__ == '__main__':
    delay(False)
    print 'Loading file'
    RRSeries = load_ds_from_csv_column("../z_data/A05.txt")
    delay()

    print 'Starting analysis'
    test(RRSeries)

    print '\n\nCache Test'
    test(RRSeries)

