from __future__ import division

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import pyPhysio as ph

#BVP
FILE = '/home/andrea/Trento/DATI/FCA/NightsAndrea/18-19_Ago/CsvOutput-CSV/data_EmpaticaDevice-Empatica E4 - A0051F/BVP.csv'
FSAMP = 64
TSTART = 0

data = np.array(pd.read_csv(FILE))
bvp = ph.EvenlySignal(data[:,1], FSAMP, 'BVP', TSTART)

#=============================
# DETECT IBI
ibi = ph.BeatFromBP()(bvp) #TODO: ibi should have the same indexes of original signal
# TODO: strange warning "data is not a Signal"

#=============================
# DETECT BAD IBI
id_bad_ibi = ph.BeatOutliers(cache = 5, sensitivity = 0.5)(ibi)

bvp.plot()
plt.vlines(ibi.get_indices(), np.min(bvp), np.max(bvp))
plt.vlines(ibi.get_indices()[id_bad_ibi], np.min(bvp), np.max(bvp), 'r')

#=============================
# OPTIMIZE IBI
ibi_opt = ph.BeatOptimizer()(ibi)/FSAMP

ax1=plt.subplot(211)
bvp.plot()
plt.vlines(ibi.get_indices(), np.min(bvp), np.max(bvp), 'r')
plt.vlines(ibi_opt.get_indices(), np.min(bvp), np.max(bvp))
plt.subplot(212, sharex=ax1)
ibi.plot('or')
ibi_opt.plot('ob')