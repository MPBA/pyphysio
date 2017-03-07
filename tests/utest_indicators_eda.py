# coding=utf-8
from __future__ import division

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import os
os.chdir('/home/andrea/Trento/CODICE/workspaces/pyHRV/pyHRV/tests')
from context import ph, Assets

#EDA
FSAMP = 4
#FILE = Asset.GSR
FILE  = '/home/andrea/Downloads/tmp/GSR.csv'
data = np.array(pd.read_csv(FILE))
eda = ph.EvenlySignal(data[:,1], FSAMP, 'EDA', 0)

eda = eda[5000*FSAMP:8000*FSAMP]
eda_f = ph.DenoiseEDA(threshold = 0.2)(eda) #OK
eda_f = eda_f.resample(8)

driver = ph.DriverEstim(T1=0.75, T2=2)(eda_f) # OK

phasic, tonic, driver_no_peak = ph.PhasicEstim(delta=0.1)(driver) # OK

mean = ph.Mean()(phasic)
std = ph.StDev()(phasic)
median = ph.Median()(phasic)
rng = ph.Range()(phasic)

mn = ph.Min()(phasic)
mx = ph.Max()(phasic)

pks_max = ph.PeaksMax(delta=0.1)(phasic)
pks_min = ph.PeaksMin(delta=0.1)(phasic)
pks_mean = ph.PeaksMean(delta=0.1)(phasic)

dur_max = ph.DurationMax(delta = 0.1, pre_max=2, post_max=2)(phasic)

slopes_min = ph.SlopeMin(delta = 0.1, pre_max=2, post_max=2)(phasic)
slopes_max = ph.SlopeMax(delta = 0.1, pre_max=2, post_max=2)(phasic)
slopes_mean = ph.SlopeMean(delta = 0.1, pre_max=2, post_max=2)(phasic)

# FAKE
data = np.zeros(80)
phasic = ph.EvenlySignal(data, 8, 'PHA', 0)

phasic[37] = 0.12
phasic[39] = 0.10
phasic[40] = 0.2
phasic[41] = 0.10
phasic[43] = 0.12

pks_max = ph.PeaksMax(delta=0.1)(phasic) #OK
pks_min = ph.PeaksMin(delta=0.1)(phasic) #OK
pks_mean = ph.PeaksMean(delta=0.1)(phasic) #OK

dur_max = ph.DurationMax(delta = 0.1, pre_max=2, post_max=2)(phasic) #OK
dur_min = ph.DurationMin(delta = 0.1, pre_max=2, post_max=2)(phasic) #OK
dur_mean = ph.DurationMean(delta = 0.1, pre_max=2, post_max=2)(phasic) #OK

slopes_min = ph.SlopeMin(delta = 0.1, pre_max=2, post_max=2)(phasic)
slopes_max = ph.SlopeMax(delta = 0.1, pre_max=2, post_max=2)(phasic)
slopes_mean = ph.SlopeMean(delta = 0.1, pre_max=2, post_max=2)(phasic)
