from __future__ import division
import numpy as np
import pandas as pd

from context import ph, Asset

#ECG
FILE = Asset.F18
data = np.array(pd.read_csv(FILE, skiprows=8, header=None))
id_ecg1 = 0
ecg1 = data[:, id_ecg1]

FSAMP = 2048
TSTART = 0

ecg = ph.EvenlySignal(ecg1, FSAMP, 'ECG', TSTART)

ibi = ph.BeatFromECG()(ecg)

# BVP
FILE = Asset.BVP
FSAMP = 64
TSTART = 0

data = np.array(pd.read_csv(FILE))
bvp = ph.EvenlySignal(data[:,1], FSAMP, 'BVP', TSTART)

ibi = ph.BeatFromBP(bpm_max=120)(bvp)


#EDA
FILE = Asset.GSR
data = np.array(pd.read_csv(FILE))
eda = ph.EvenlySignal(data[:,1], 4, 'EDA', 0)

eda_f = ph.DenoiseEDA(threshold = 0.2)(eda)
eda_f = eda_f.resample(8)

driver = ph.DriverEstim(T1=0.75, T2=2)(eda_f)

phasic, tonic, driver_no_peak = ph.PhasicEstim(delta=0.1)(driver)
