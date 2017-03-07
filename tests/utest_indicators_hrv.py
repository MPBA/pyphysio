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
# TEST IBI EXTRACTION FROM ECG
ecg = ph.EvenlySignal(data[:, 0], sampling_freq = FSAMP, signal_nature = 'ECG', start_time = TSTART)

ecg = ecg.resample(fout=4096, kind='cubic')

ibi = ph.BeatFromECG()(ecg)
#%%

assert(int(ph.Mean()(ibi)*10000) == 8619)
assert(int(ph.StDev()(ibi)*10000) == 602)

assert(int(ph.Median()(ibi)*10000) == 8679)
assert(int(ph.Range()(ibi)*10000) == 2548)

assert(int(ph.RMSSD()(ibi)*10000) == 328)
assert(int(ph.SDSD()(ibi)*10000) == 328)

# TODO: almost equal below
assert(int(ph.PowerInBand(interp_freq=4, method = 'welch', freq_max=0.04, freq_min=0.00001)(ibi)*10000) == 1271328)
assert(int(ph.PowerInBand(interp_freq=4, method = 'welch', freq_max=0.15, freq_min=0.04)(ibi)*10000) == 2599850)
assert(int(ph.PowerInBand(method = 'welch', interp_freq=4, freq_max=0.4, freq_min=0.15)(ibi)*10000) == 1201839)


assert(int(ph.PNNx(threshold=10)(ibi)*10000) == 3453)
assert(int(ph.PNNx(threshold=25)(ibi)*10000) == 2158)
assert(int(ph.PNNx(threshold=50)(ibi)*10000) == 431)

#%%
# Test with FAKE IBI
idx_ibi = np.arange(0, 101, 10).astype(float)
ibi = ph.UnevenlySignal(np.diff(idx_ibi), idx_ibi[1:], 10, 90, 'IBI')
ibi[-1] = 10.011
mean = ph.Mean()(ibi)  # OK
std = ph.StDev()(ibi)  # OK
median = ph.Median()(ibi)  # OK
rng = ph.Range()(ibi)  # OK

VLF = ph.PowerInBand(interp_freq=4, freq_max=0.04, freq_min=0.00001)(ibi) # OK
LF = ph.PowerInBand(interp_freq=4, freq_max=0.15, freq_min=0.04)(ibi) # OK
HF = ph.PowerInBand(interp_freq=4, freq_max=0.4, freq_min=0.15)(ibi) # OK

rmssd = ph.RMSSD()(ibi)  # OK
sdsd = ph.SDSD()(ibi)  # OK

pnn10 = ph.PNNx(threshold=10)(ibi) # OK
pnn25 = ph.PNNx(threshold=25)(ibi) # OK
pnn50 = ph.PNNx(threshold=50)(ibi) # OK

mn = ph.Min()(ibi)
mx = ph.Max()(ibi)

sm = ph.Sum()(ibi)

window_generator = ph.TimeSegments(step=30, width=60)
windows_new = window_generator(ibi)

indicators = [ph.Mean(), ph.StDev(), ph.Median(), ph.Range(), ph.StDev(), ph.RMSSD(), ph.SDSD(), ph.TINN(), ph.PowerInBand(interp_freq=4, freq_max=0.04, freq_min=0.00001), ph.PowerInBand(interp_freq=4, freq_max=0.15, freq_min=0.04), ph.PowerInBand(interp_freq=4, freq_max=0.4, freq_min=0.15), ph.PNNx(threshold=10), ph.PNNx(threshold=25), ph.PNNx(threshold=50)]

results, labels, col_names = ph.fmap(windows_new, indicators) #TODO: return results (matrix of indicators), labels (column of labels), column names
# TODO: add name of indicator as parameter: es mean_hrv = ph.Mean('RRmean'), mean_eda = ph.Mean('EDAmean'), HF = ph.PowerInBand(interp_freq=4, freq_max=0.4, freq_min=0.15)(ibi) # OK to be returned as column name
