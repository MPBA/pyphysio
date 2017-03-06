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
ecg = ph.EvenlySignal(data[:, 0], sampling_freq = FSAMP, signal_nature = 'ECG', start_time = TSTART)

#%%
# TEST Normalize
# TODO: implement below as almostequal
assert(np.mean(ph.Normalize(norm_method='standard')(ecg))==0)  # Almost equal

assert(np.min(ph.Normalize(norm_method='min')(ecg)) == 0)  # OK

assert(np.min(ph.Normalize(norm_method='maxmin')(ecg)) == 0)  # OK
assert(np.max(ph.Normalize(norm_method='maxmin')(ecg)) == 1)  # OK

ecg_ = ph.Normalize(norm_method='custom', norm_bias=4, norm_range=0.1)(ecg)
assert(np.max(ecg_) - np.min(ecg_) == 16.41)

#%%
# TEST Diff
s = ph.EvenlySignal(np.arange(1000), sampling_freq = FSAMP, signal_nature = '', start_time = TSTART)
#degree = [5, 50, -1]
degree = 5
assert(np.mean(ph.Diff(degree=degree)(s)) == degree) # OK

#%% TEST IIRFilter
assert(int(np.max(ph.IIRFilter(fp=10, fs=70)(ecg))*10000) == 8238)
assert(int(np.max(ph.IIRFilter(fp=70, fs=45)(ecg))*10000) == 2144)
assert(int(np.max(ph.IIRFilter(fp=[5, 25], fs=[0.05, 50], ftype='ellip')(ecg))*10000) == 8786)

#%%

# TEST ConvolutionalFilter
assert(int(np.max(ph.ConvolutionalFilter(irftype='gauss', win_len=0.1)(ecg))*10000) == 7501)
assert(int(np.max(ph.ConvolutionalFilter(irftype='rect', win_len=0.1)(ecg))*10000) == 4022)
assert(int(np.max(ph.ConvolutionalFilter(irftype='triang', win_len=0.1)(ecg))*10000) == 5000)
assert(int(np.max(ph.ConvolutionalFilter(irftype='dgauss', win_len=0.1, normalize=False)(ecg))*10000) == 8823)

irf = np.r_[np.arange(50,500, 5)]
ecg_cfC = ph.ConvolutionalFilter(irftype='custom', irf = irf, normalize=True)(ecg)
assert(int(np.max(ecg_cfC)*10000) == 7723)

#%%
# TEST DeConvolutionalFilter 
ecg_df = ph.DeConvolutionalFilter(irf=irf, normalize=True, deconv_method='fft')(ecg_cfC)  # OK
assert(int(np.max(ecg_df)*10000) == 48973)
