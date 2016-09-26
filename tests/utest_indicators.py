from __future__ import division

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from context import ph, Asset

FILE = Asset.F18

data = np.array(pd.read_csv(FILE, skiprows=8, header=None))

id_ecg1 = 0
id_ecg2 = 6

ecg1 = data[:, id_ecg1]

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
ecg.plot()
plt.vlines(ibi.get_indices(), np.min(ecg), np.max(ecg))

mean = ph.Mean()(ibi)
std = ph.StDev()(ibi)
median = ph.Median()(ibi)
rng = ph.Range()(ibi)
rmssd = ph.RMSSD()(ibi)
sdsd = ph.SDSD()(ibi)
# TODO (Andrea): Variables contain tuples because of the ending comma of the line
VLF = ph.PowerInBand(interp_freq=4, freq_max=0.04, freq_min=0.00001)(ibi),  # TODO: Should return a scalar
LF = ph.PowerInBand(interp_freq=4, freq_max=0.15, freq_min=0.04)(ibi),  # TODO: Should return a scalar
HF = ph.PowerInBand(interp_freq=4, freq_max=0.4, freq_min=0.15)(ibi),  # TODO: Should return a scalar

pnn10 = ph.PNNx(threshold=10)(ibi)
pnn25 = ph.PNNx(threshold=25)(ibi)
pnn50 = ph.PNNx(threshold=50)(ibi)

# FAKE IBI
idx_ibi = np.arange(0, 100, 10).astype(float)
ibi = ph.UnevenlySignal(np.diff(idx_ibi), idx_ibi[1:], 10, 90, 'IBI')
ibi[-1] = 10.011
mean = ph.Mean()(ibi)  # OK
std = ph.StDev()(ibi)  # OK
median = ph.Median()(ibi)  # OK
rng = ph.Range()(ibi)  # OK

VLF = ph.PowerInBand(interp_freq=4, freq_max=0.04, freq_min=0.00001)(ibi),  # TODO: Should return a scalar
LF = ph.PowerInBand(interp_freq=4, freq_max=0.15, freq_min=0.04)(ibi),  # TODO: Should return a scalar
HF = ph.PowerInBand(interp_freq=4, freq_max=0.4, freq_min=0.15)(ibi),  # TODO: Should return a scalar

rmssd = ph.RMSSD()(ibi)  # OK
sdsd = ph.SDSD()(ibi)  # OK

pnn10 = ph.PNNx(threshold=10)(ibi)
pnn25 = ph.PNNx(threshold=25)(ibi)
pnn50 = ph.PNNx(threshold=50)(ibi)

mn = ph.Min()(ibi)
mx = ph.Max()(ibi)

sm = ph.Sum()(ibi)

AUC = ph.AUC()(ibi)
