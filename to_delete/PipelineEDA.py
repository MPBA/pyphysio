# coding=utf-8
from __future__ import division

import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt

import pyphysio.Filters as flt_old
import pyphysio.Tools as tll_old
import pyphysio.Estimators as est_old
import pyphysio.Segmentation as sgm_old
import pyphysio.Indexes as ind_old

from pyphysio.Signal import EvenlySignal as sig
from pyphysio.filters import  Filters as flt_new
from pyphysio.tools import Tools as tll_new
from pyphysio.estimators import Estimators as est_new
from pyphysio.segmentation import SegmentsGenerators as sgm_new
from pyphysio.indicators import TimeDomain as td_new
from pyphysio.indicators import FrequencyDomain as fd_new
from pyphysio.indicators import NonLinearDomain as nl_new
import pyphysio as ph

FILE = '/home/andrea/Trento/DATI/SYNC/F/F18.txt'

data = np.array(pd.read_csv(FILE, skiprows=8, header=None))

id_eda1 = 1
id_eda2 = 3
id_emg1 = 4
id_emg2 = 7
id_ecg1 = 5
id_ecg2 = 6
id_tt = 2

eda1 = data[:, id_eda1]

fsamp = 2048

#===========================
# initialize signals
# OLD
eda_np = eda1
# NEW
eda_pp = sig(eda1, fsamp, 'EDA')

#===========================
# RESAMPLE
fout = 128
# OLD
eda_np_res = flt_old.resample(eda_np, fsamp, fout)
# NEW
eda_pp_res = eda_pp.resample(fout)

print(np.unique(eda_np_res - eda_pp_res)) # 0

eda_np = eda_np_res
eda_pp = eda_pp_res
fsamp = 128

#===========================
# SMOOTHING
# OLD
eda_np_smt = flt_old.convolutional_filter(eda_np, fsamp, wintype = 'gauss', winsize = 2)
# NEW
conv_filter = flt_new.ConvolutionalFilter(irftype = 'gauss', win_len = 2)
eda_pp_smt = conv_filter(eda_pp)

print(np.unique(eda_np_smt - eda_pp_smt)) # 0

eda_np = eda_np_smt
eda_pp = eda_pp_smt

#===========================
# FILTERING LOW PASS
fp = 20
fs = 25
# OLD
b, a = flt_old.iir_coefficients(fp, fs, fsamp, plot=False)
eda_np_flt = flt_old.iir_filter(eda_np, b, a)
# NEW
f_20_25 = flt_new.IIRFilter(fp = fp, fs = fs)
eda_pp_flt = f_20_25(eda_pp)

eda_np = eda_np_flt
eda_pp = eda_pp_flt

#===========================
# RESAMPLE
fout = 16
# OLD
eda_np_res = flt_old.resample(eda_np, fsamp, fout)
# NEW
eda_pp_res = eda_pp.resample(fout)

print(np.unique(eda_np_res - eda_pp_res)) # 0

eda_np = eda_np_res
eda_pp = eda_pp_res
fsamp = 16

#==============================
# ESTIMATE DRIVER
# custom params
par_bat = [0.75, 2]
# OLD
driver_np = est_old.estimate_driver(eda_np, fsamp, par_bat)
# NEW
driver_estimator = est_new.DriverEstim(T1 = 0.75, T2 = 2)
driver_pp = driver_estimator(eda_pp)
# TODO (Ale) Default value in DeConvolutionalFilter for normalize: None -> dovrebbe essere True

# optimized params
delta = 0.01
# OLD
#par_bat_old = est_old.optimize_bateman_simple(eda_np, fsamp, 'asa', delta, verbose=True, maxiter=50)
#par_bat_old = [ 0.51004343, 5.3033562 ]
#driver_np = est_old.estimate_driver(eda_np, fsamp, par_bat_old)
# NEW
par_bat_estimator = tll_new.OptimizeBateman(opt_method='asa', complete=True, pars_ranges=[0.01, 1, 1, 15], maxiter=50, delta=delta)
# T1 = 0.098
# T2 = 8.567
par_bat_new = par_bat_estimator(eda_pp)
T1, T2 = par_bat_new[0]
driver_estimator = est_new.DriverEstim(T1 = T1, T2 = T2)
driver_pp = driver_estimator(eda_pp)

#==============================
# ESTIMATE PHASIC COMPONENT
# OLD
phasic_np, tonic_np, driver_no_peak_np = est_old.estimate_components(driver_np, fsamp, delta, 1, fsamp, fsamp)
#NEW
phasic_estimator = est_new.PhasicEstim(delta=delta, grid_size = 1, pre_max = 1, post_max = 1)
phasic_pp, tonic_pp, driver_no_peak_pp = phasic_estimator(driver_pp)

#=============================
# GENERATE WINDOWS
#OLD
windows_old, labels = sgm_old.get_windows_contiguos(np.zeros(len(phasic_np)), 60*fsamp, 30*fsamp)
#NEW
window_generator = sgm_new.LengthSegments(step = 30, width = 60)
windows_new = window_generator(phasic_pp)

#============================
# COMPUTE INDICATORS
#OLD
#feat_1, label_col_1 = sgm_old.compute_on_windows(ibi_old/fsamp, fsamp, windows, ind_old.HRVfeatures, **{'method':'welch'})
#NEW

algos = [td_new.Mean(),
         td_new.StDev(),
         td_new.Median(),
         td_new.Range(),
         td_new.StDev(),
         td_new.RMSSD(),
         td_new.SDSD(),
         td_new.TINN(),
#         fd_new.PowerInBand(interp_freq=4, freq_max=0.04, freq_min=0.00001), #VLF
#         fd_new.PowerInBand(interp_freq=4, freq_max=0.15, freq_min=0.04), #LF
#         fd_new.PowerInBand(interp_freq=4, freq_max=0.4, freq_min=0.15), #HF
         nl_new.PNNx(threshold=10), #PNN10
         nl_new.PNNx(threshold=25), #PNN25
         nl_new.PNNx(threshold=50) #PNN50
         ]

results = ph.fmap(windows_new, algos)
    
