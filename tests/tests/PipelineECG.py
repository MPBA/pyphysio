from __future__ import division

import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt

import pyphysio.Filters as flt_old
import pyphysio.Tools as tll_old
import pyphysio.Estimators as est_old

PYPHYSIODIR = '/home/andrea/Trento/CODICE/workspaces/pyHRV-AleB/pyHRV/pyHRV'

os.chdir(PYPHYSIODIR)
from pyPhysio.Signal import EvenlySignal as sig
from pyPhysio.filters import  Filters as flt_new
from pyPhysio.tools import Tools as tll_new
from pyPhysio.estimators import Estimators as est_new

FILE = '/home/andrea/Trento/DATI/SYNC/F/F18.txt'

data = np.array(pd.read_csv(FILE, skiprows=8, header=None))

if np.shape(data)[1] == 10:
    labels = ['time','trgM','edaM','tt','edaF','trgF','emgM','ecgM','ecg2F','emgF']
    id_trg1 = 1
    id_trg2 = 5
    id_eda1 = 2
    id_eda2 = 4
    id_emg1 = 6
    id_emg2 = 9
    id_ecg1 = 7
    id_ecg2 = 8
    id_tt = 3
elif np.shape(data)[1] == 8:
    labels = ['time','eda1','tt','eda2','emg1','ecg1','ecg2','emg2']
    id_eda1 = 1
    id_eda2 = 3
    id_emg1 = 4
    id_emg2 = 7
    id_ecg1 = 5
    id_ecg2 = 6
    id_tt = 2

ecg1 = data[:, id_ecg1]

fsamp = 2048

#===========================
# initialize signals
# OLD
ecg_np = ecg1[100000: 200000]
# NEW
ecg_pp = sig(ecg_np, fsamp, 'ECG')

#===========================
# RESAMPLE
fout = 128
# OLD
ecg_np_res = flt_old.resample(ecg_np, fsamp, fout)
# NEW
ecg_pp_res = ecg_pp.resample(fout)

#print(np.unique(ecg_np_res - ecg_pp_res)) # 0

ecg_np = ecg_np_res
ecg_pp = ecg_pp_res
fsamp = 128

#===========================
# FILTERING
fp = 25
fs = 35
# OLD
b, a = flt_old.iir_coefficients(fp, fs, fsamp, plot=False)
ecg_np_flt = flt_old.iir_filter(ecg_np, b, a)
# NEW
f_25_35 = flt_new.IIRFilter(fp = fp, fs = fs)
ecg_pp_flt = f_25_35(ecg_pp) 
#FIXME: ecg_pp_flt NON E' PIU' UN SIGNAL !!!!
ecg_pp_flt = sig(ecg_pp_flt, fsamp, 'ECG')

# filtering with bad settings
fp = 25
fs = 26
f_25_26 = flt_new.IIRFilter(fp = fp, fs = fs)
ecg_pp_flt = f_25_26(ecg_pp)
ecg_pp_flt = f_25_26(ecg_pp) # La seconda volta non mi da il warning
#

ecg_pp_flt = f_25_35(ecg_pp)

ecg_np = ecg_np_flt
ecg_pp = ecg_pp_flt

#=============================
# ESTIMATE DELTA
# OLD
#range_ecg_old = tll_old.estimate_delta(ecg_np, fsamp/2, fsamp, gauss_filt=False)
#range_ecg_old = np.median(range_ecg_old*0.7)
# NEW
range_estimator = tll_new.SignalRange(win_len = 1, win_step = 0.5, smooth = False)
#ERRORE: Voleva parametri int, in realta' sono float. Corretto
range_ecg_new = range_estimator(ecg_pp)
range_ecg_new = np.median(range_ecg_new*0.7)

#print(np.unique((range_ecg_old - range_ecg_new)))

#=============================
# PEAK DETECTION
# OLD
#idx_ibi_old, ibi_old = est_old.estimate_peaks_ecg(ecg_np, fsamp, range_ecg_old)
# NEW

# STRANI ERRORI
ibi_estimator = est_new.BeatFromECG() #FIXME: Non da errore ma il default per bpm_max=1, dovrebbe essere 180

ibi_estimator = est_new.BeatFromECG(delta = range_ecg_new) # OK usa il default (comunque sbagliato) per bpm_max

ibi_estimator = est_new.BeatFromECG(bpm_max = 180) #FIXME: ERRORE di difficile interpretazione


# Provo comunque:
ibi_estimator = est_new.BeatFromECG()
ibi = ibi_estimator(ecg_pp) #FIXME: errore in pyPhysio/tools/Tools.py", line 244