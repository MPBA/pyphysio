from __future__ import division

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from context import ph, Asset

FILE = Asset.F18

data = np.array(pd.read_csv(FILE, skiprows=8, header=None))

ecg1 = data[:, 0]

fsamp = 2048

# ===========================
# INITIALIZING
ecg = ph.EvenlySignal(ecg1, fsamp, 'ECG', 0)
ecg = ecg[1000000:1500000]
ecg.plot()

# =============================
# ESTIMATE DELTA
range_ecg = ph.SignalRange(win_len=2, win_step=0.5, smooth=False)(ecg)

# =============================
# DETECT IBI
ibi = ph.BeatFromECG(deltas=range_ecg * 0.7)(ecg)

# PLOT
plt.plot(ecg)
plt.vlines(ibi.get_indices(), np.min(ecg), np.max(ecg))

mean = ph.Mean()(ibi)
std = ph.StDev()(ibi)
median = ph.Median()(ibi)
rng = ph.Range()(ibi)
rmssd = ph.RMSSD()(ibi)
sdsd = ph.SDSD()(ibi)
VLF = ph.PowerInBand(interp_freq=4, freq_max=0.04, freq_min=0.00001)(ibi)
LF = ph.PowerInBand(interp_freq=4, freq_max=0.15, freq_min=0.04)(ibi)
HF = ph.PowerInBand(interp_freq=4, freq_max=0.4, freq_min=0.15)(ibi)

pnn10 = ph.PNNx(threshold=10)(ibi)
pnn25 = ph.PNNx(threshold=25)(ibi)
pnn50 = ph.PNNx(threshold=50)(ibi)

# FAKE IBI
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

results, col_names = ph.fmap(windows_new, indicators) #TODO: return results (matrix of indicators), labels (column of labels), column names
