# coding=utf-8
from __future__ import division

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import pyphysio as ph

# BVP
FILE = '/home/andrea/Trento/DATI/Dataset_ABP/experiment/sub01//BVP.csv'
FSAMP = 64
TSTART = 0

data = np.array(pd.read_csv(FILE))
bvp = ph.EvenlySignal(data[:, 1], FSAMP, 'BVP', TSTART)

ibi = ph.BeatFromBP()(bvp)  





# =============================
# DETECT IBI
ibi = ph.BeatFromBP()(bvp)  # TODO: ibi should have the same indexes of original signal /// it does, try this:
# plt.plot(bvp)
# plt.plot(ibi.get_indices(), bvp[np.asarray(ibi.get_indices(), 'i')], 'ro')
# plt.show()
# TODO: strange warning "data is not a Signal" ///
# that's because of the Diff()(x) where x is not a signal in the implementations, _np.diff should be used instead
# the last commit fixed this issue

# =============================
# DETECT BAD IBI
id_bad_ibi = ph.BeatOutliers(cache=5, sensitivity=0.5)(ibi)

bvp.plot()
plt.vlines(ibi.get_indices(), np.min(bvp), np.max(bvp))
plt.vlines(ibi.get_indices()[id_bad_ibi], np.min(bvp), np.max(bvp), 'r')

# =============================
# OPTIMIZE IBI
ibi_opt = ph.BeatOptimizer()(ibi) / FSAMP

ax1 = plt.subplot(211)
bvp.plot()
plt.vlines(ibi.get_indices(), np.min(bvp), np.max(bvp), 'r')
plt.vlines(ibi_opt.get_indices(), np.min(bvp), np.max(bvp))
plt.subplot(212, sharex=ax1)
ibi.plot('or')
ibi_opt.plot('ob')

g = ph.FixedSegments(width=60, step=10)(ibi)

h = ph.PowerInBand(method="ar", interp_freq = 4, freq_min=1, freq_max=3)

values, labels, colnames = ph.fmap(g, [h])

plt.plot(values[:,2])
plt.show()