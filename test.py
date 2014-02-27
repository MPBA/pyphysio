import time
import pyhrv
import rr

_debug_time = 0


def delay(v=True):
    global _debug_time
    d = time.time() - _debug_time
    _debug_time = time.time()
    if v:
        print 'Delay: ', int(d * 100000) / 100.0, 'ms'


def test(RRseries):
    print '1. TD'
    TD_ind, TD_lab = rr.RRAnalysis.TD_indexes(RRseries)
    print TD_lab
    print TD_ind
    delay()
    print '2. POIN'
    POIN_ind, POIN_lab = rr.RRAnalysis.poin_indexes(RRseries)
    print POIN_lab
    print POIN_ind
    delay()
    print '3. FD'
    FD_ind, FD_lab = rr.RRAnalysis.FD_indexes(RRseries, 1)
    print FD_lab
    print FD_ind
    delay()

if __name__ == '__main__':
    delay(False)

    RRseries = pyhrv.DataSeries.from_csv_ibi_or_rr("Subject_data/B01.txt")

    print 'Starting analysis'
    test(RRseries)

    print 'Cache test'
    test(RRseries)
