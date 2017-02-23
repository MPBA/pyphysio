# coding=utf-8
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
ecg.plot()

# ===========================
# FILTERING LOW PASS
fp = 50
fs = 80

ecg_f = ph.IIRFilter(fp=[47, 53], fs=[49.5, 50.5])(ecg)
ecg = ph.IIRFilter(fp=fp, fs=fs)(ecg_f)

# =============================
# ESTIMATE DELTA
range_estimator = ph.SignalRange(win_len=2, win_step=0.5, smooth=False)(ecg)

# =============================
# DETECT IBI
ibi = ph.BeatFromECG(deltas=range_estimator * 0.7)(ecg)

# =============================
# DETECT BAD IBI
id_bad_ibi = ph.BeatOutliers(cache=5, sensitivity=0.5)(ibi)

ecg.plot()
plt.vlines(ibi.get_indices(), np.min(ecg), np.max(ecg))
plt.vlines(ibi.get_indices()[id_bad_ibi], np.min(ecg), np.max(ecg), 'r')

# =============================
# FIX IBI
ibi = ph.FixIBI(id_bad_ibi=id_bad_ibi)(ibi)

# PLOT
ecg.plot()
plt.vlines(ibi.get_indices(), np.min(ecg), np.max(ecg))

# =============================
# WINDOWING
windows = ph.LengthSegments(step=30, width=60)(ibi)

# ============================
# COMPUTE INDICATORS
algos = [ph.Mean(),
         ph.StDev(),
         ph.Median(),
         ph.Range(),
         ph.RMSSD(),
         ph.SDSD(),
         ph.TINN(),
         ph.PowerInBand(interp_freq=4, freq_max=0.04, freq_min=0.00001),  # VLF
         ph.PowerInBand(interp_freq=4, freq_max=0.15, freq_min=0.04),  # LF
         ph.PowerInBand(interp_freq=4, freq_max=0.4, freq_min=0.15),  # HF
         ph.PNNx(threshold=10),  # PNN10
         ph.PNNx(threshold=25),  # PNN25
         ph.PNNx(threshold=50)  # PNN50
         ]

results = ph.fmap(windows, algos)  # TODO: return numpy array and column names /// Done, there is no other way to return
# the parameters, we can change the syntax but the name and the value should be specified
