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
#EDA
eda = ph.EvenlySignal(data[:,1], sampling_freq = FSAMP, signal_nature = 'EDA', start_time = 0)

#preprocessing
## downsampling
eda_r = eda.resample(4)
    
## filter
eda = ph.IIRFilter(fp=0.8, fs=1.1)(eda_r)

#%%
#estimate driver
driver = ph.DriverEstim(T1=0.75, T2=2)(eda)
assert(int(np.mean(driver)*10000) == 18255)
assert(isinstance(driver, ph.EvenlySignal))

#%%
phasic, tonic, driver_no_peak = ph.PhasicEstim(delta=0.1)(driver)
assert(int(np.mean(phasic)*10000) == 485)
assert(isinstance(phasic, ph.EvenlySignal))