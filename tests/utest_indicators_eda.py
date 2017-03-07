# coding=utf-8
from __future__ import division

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import os
os.chdir('/home/andrea/Trento/CODICE/workspaces/pyHRV/pyHRV/tests')
from context import ph, Assets

#%%
#EDA
FSAMP = 2048

eda = ph.EvenlySignal(Assets.eda(), sampling_freq = FSAMP, signal_nature = 'EDA', start_time = 0)

#preprocessing
## downsampling
eda = eda.resample(8)
    
## filter
eda = ph.IIRFilter(fp=0.8, fs=1.1)(eda)

#estimate driver
driver = ph.DriverEstim(T1=0.75, T2=2)(eda)
phasic, tonic, driver_no_peak = ph.PhasicEstim(delta=0.1)(driver)

#%%
assert(int(ph.Mean()(phasic)*10000) == 575)

mx = ph.Max()(phasic)
pks_max = ph.PeaksMax(delta=0.1)(phasic)

assert(pks_max == mx)


idx_mx, idx_mn, mx, mn = ph.PeakDetection(delta = 0.1)(phasic)

n_peaks = ph.PeaksNum(delta = 0.1, pre_max=2, post_max=2)(phasic)
assert(n_peaks == 24)

st, sp = ph.PeakSelection(maxs=idx_mx, pre_max=3, post_max=5)(phasic)
assert(np.sum(st)== 11519)
assert(np.sum(sp)== 12210)

dur_max = ph.DurationMax(delta = 0.1, pre_max=2, post_max=2)(phasic)

#%%
# FAKE PHASIC
data = np.zeros(80)
phasic = ph.EvenlySignal(data, sampling_freq = 8, signal_nature = 'PHA', start_time = 0)

phasic[37] = 0.12
phasic[39] = 0.10
phasic[40] = 0.2
phasic[41] = 0.10
phasic[43] = 0.12

pks_max = ph.PeaksMax(delta=0.1)(phasic)
assert(pks_max == 0.2)

pks_min = ph.PeaksMin(delta=0.1)(phasic)
assert(pks_min == 0.12)

pks_mean = ph.PeaksMean(delta=0.1)(phasic)
assert(pks_mean == np.mean([0.12, 0.2, 0.12]))

dur_max = ph.DurationMax(delta = 0.1, pre_max=2, post_max=2)(phasic) #OK
assert(dur_max == 0.5)

dur_min = ph.DurationMin(delta = 0.1, pre_max=2, post_max=2)(phasic) #OK
assert(dur_min == 0.25)

dur_mean = ph.DurationMean(delta = 0.1, pre_max=2, post_max=2)(phasic) #OK
assert(dur_mean == np.mean([0.25, 0.25, 0.5]))
