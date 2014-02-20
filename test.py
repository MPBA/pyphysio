import numpy as np
from pyhrv import *
from utility import *

np.random.seed()

t = np.arange(0, 10*np.pi, 0.01*np.pi)
RRraw = np.random.uniform(500, 1500, 1000)+100*np.sin(t)

RRseries = DataSeries(RRraw)

print 'Starting analysis'

print '.TD'
TD_ind, TD_lab = RRAnalysis.TD_indexes(RRseries)
print TD_lab
print TD_ind
print '.POIN'
POIN_ind, POIN_lab = RRAnalysis.POIN_indexes(RRseries)
print POIN_lab
print POIN_ind
print '.FD'
FD_ind, FD_lab = RRAnalysis.FD_indexes(RRseries, 1)
print FD_lab
print FD_ind
