# coding=utf-8
from __future__ import division
import numpy as np
import pandas as pd

import pyphysio as ph

# TODO: rewrite the code in appropriate format for tests

#%%
FILE = '/home/andrea/Trento/CODICE/workspaces/pyHRV/pyHRV/sample_data/medical.txt'
FSAMP = 2048
TSTART = 0

data = np.loadtxt(FILE, delimiter='\t')

#%%
bvp = ph.EvenlySignal(data[:, 2], sampling_freq = FSAMP, signal_nature = 'ECG', start_time = TSTART)

#%%
# TEST SignalRange
rng_bvp = ph.SignalRange(win_len=1, win_step=0.5, smooth=True)(bvp)  # OK
assert(int(np.max(rng_bvp)*100) == 1746)

#%%
# TEST PeakDetection
idx_mx, idx_mn, mx, mn = ph.PeakDetection(delta=rng_bvp * 0.5)(bvp)

assert(np.sum(idx_mx)==16818171)
assert(np.sum(idx_mn)==16726017)

#%%
# EDA
eda = ph.EvenlySignal(data[:, 1], sampling_freq = FSAMP, signal_nature = 'EDA', start_time = TSTART)
eda = eda.resample(fout = 8)

eda = ph.ConvolutionalFilter(irftype='gauss', win_len=2)(eda)

driver = ph.DriverEstim(delta=0.02)(eda)
phasic, _, __ = ph.PhasicEstim(delta=0.02)(driver)

idx_mx, idx_mn, mx, mn = ph.PeakDetection(delta=0.02)(phasic)


# TEST PeakSelection
st, sp = ph.PeakSelection(maxs=idx_mx, pre_max=3, post_max=5)(phasic)
assert(np.sum(st)==21244)

#%%
# TEST PSD
FSAMP = 100
n = np.arange(1000)
t = n / FSAMP
freq = 2.5

sinusoid = ph.EvenlySignal(np.sin(2 * np.pi * freq * t), sampling_freq = FSAMP, signal_nature = '', start_time = 0)
#sinusoid.plot()

f, psd = ph.PSD(method='welch', nfft=2048, window='hanning')(sinusoid)
#TODO AlmostEqual below
assert(f[np.argmax(psd)]==2.5)


sinusoid_unevenly = ph.UnevenlySignal(np.delete(sinusoid.get_values(), [10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]),
                                      sampling_freq = FSAMP, signal_nature = '', start_time = 0,
                                      x_values = np.delete(sinusoid.get_times(), [10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]),
                                      x_type='instants')
#sinusoid_unevenly.plot('.-')

sinusoid_unevenly = sinusoid_unevenly.to_evenly(kind='linear') #should be possible to remove this line after Issue #24 is solved

f, psd = ph.PSD(method='welch', nfft=2048, window='hanning')(sinusoid_unevenly)

#TODO AlmostEqual below
assert(f[np.argmax(psd)]==2.5)

#%%
# TEST Maxima
idx_mx, mx = ph.Maxima(method='windowing', win_len=1, win_step=0.5)(bvp)
assert(np.sum(idx_mx)==16923339)

# TEST Minima
idx_mn, mn = ph.Minima(method='windowing', win_len=1, win_step=0.5)(bvp)
assert(np.sum(idx_mn)==17939276)