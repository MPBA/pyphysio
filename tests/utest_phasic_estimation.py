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

eda_f = ph.DenoiseEDA(threshold = 0.2)(eda)
eda_f = eda_f.resample(8)

driver = ph.DriverEstim(T1=0.75, T2=2)(eda_f)

phasic, tonic, driver_no_peak = ph.PhasicEstim(delta=0.1)(driver)