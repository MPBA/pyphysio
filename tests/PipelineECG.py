# coding=utf-8
from __future__ import division

from context import ph, Asset
import numpy as np
import pandas as pd

FILE = Asset.F18

data = np.array(pd.read_csv(FILE, skiprows=8, header=None))

id_eda1 = 1
id_eda2 = 3
id_emg1 = 4
id_emg2 = 7
id_ecg1 = 5
id_ecg2 = 6
id_tt = 2

ecg1 = data[:, id_ecg1]

fsamp = 2048

# ===========================
# initialize signals
# OLD
# ecg_np = ecg1
# NEW
ecg_pp = ph.Signal(ecg1, fsamp, 'ECG')

ecg_pp.plot()
# ===========================
# RESAMPLE
fout = 128
# OLD
# ecg_np_res = flt_old.resample(ecg_np, fsamp, fout)
# NEW
ecg_pp = ecg_pp.resample(fout)

# print(np.unique(ecg_np_res - ecg_pp_res)) # 0

# ecg_np = ecg_np_res
# ecg_pp = ecg_pp_res
fsamp = 128

# ===========================
# FILTERING LOW PASS
fp = 25
fs = 30
# OLD
# b, a = flt_old.iir_coefficients(fp, fs, fsamp, plot=True)
# ecg_np_flt = flt_old.iir_filter(ecg_np, b, a)
# NEW
f_25_35 = ph.IIRFilter(fp=fp, fs=fs)
ecg_pp = f_25_35(ecg_pp)

# FILTERING HIGH PASS
fp = 5
fs = 1
# b, a = flt_old.iir_coefficients(fp, fs, fsamp, plot=False)
# ecg_np_flt = flt_old.iir_filter(ecg_np_flt, b, a)

f_5_1 = ph.IIRFilter(fp=fp, fs=fs)
ecg_pp = f_5_1(ecg_pp)

##=============================
## ESTIMATE DELTA
## OLD
# range_ecg_old = tll_old.estimate_delta(ecg_np, fsamp/2, 2*fsamp, gauss_filt=False)
# range_ecg_old = np.median(range_ecg_old*0.7)
## NEW
# range_estimator = tll_new.SignalRange(win_len = 2, win_step = 0.5, smooth = False)
# range_ecg_new = range_estimator(ecg_pp)
# range_ecg_new = np.median(range_ecg_new*0.7)
#
##print(np.unique((range_ecg_old - range_ecg_new))) # 0 OK

# =============================
# PEAK DETECTION
# OLD
# idx_ibi_old, ibi_old = est_old.estimate_peaks_ecg(ecg_np, fsamp, range_ecg_old*0.7)
# NEW
ibi_estimator = ph.BeatFromECG()

ibi = ibi_estimator(ecg_pp)  # FIXME: "The data is not a Signal." Falso
# Mi dava errore controlla codice

# print(np.unique(ibi_old/fsamp - ibi.get_y_values())) # 0 OK

# set ibi for old processing
# ibi_old_ = np.repeat(np.nan, len(ecg_np))
# ibi_old_[idx_ibi_old.astype(int)] = ibi_old
# ibi_old = ibi_old_

# =============================
# GENERATE WINDOWS
# OLD
# windows_old, labels = sgm_old.get_windows_contiguos(np.zeros(len(ibi_old)), 60*fsamp, 30*fsamp)
# NEW
window_generator = ph.LengthSegments(step=30, width=60)
windows_new = window_generator(ibi)

# ============================
# COMPUTE INDICATORS
# OLD
# feat_1, label_col_1 = sgm_old.compute_on_windows(ibi_old/fsamp, fsamp, windows, ind_old.HRVfeatures, **{'method':'welch'})
# NEW

algos = [
            ph.Mean(),
            ph.StDev(),
            ph.Median(),
            ph.Range(),
            ph.StDev(),
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

results = ph.fmap(windows_new, algos)
