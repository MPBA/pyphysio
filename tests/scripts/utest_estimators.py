from __future__ import division
import numpy as np
import pandas as pd

import pyPhysio as ph

#ECG
FILE = '/home/andrea/Trento/DATI/SYNC/F/F18.txt'
data = np.array(pd.read_csv(FILE, skiprows=8, header=None))
id_ecg1 = 5
ecg1 = data[:, id_ecg1]

FSAMP = 2048
TSTART = 0

ecg = ph.EvenlySignal(ecg1, FSAMP, 'ECG', TSTART) # OK

ibi = ph.BeatFromECG()(ecg)

#BVP
FILE = '/home/andrea/Trento/DATI/FCA/NightsAndrea/18-19_Ago/CsvOutput-CSV/data_EmpaticaDevice-Empatica E4 - A0051F/BVP.csv'
FSAMP = 64
TSTART = 0

data = np.array(pd.read_csv(FILE))
bvp = ph.EvenlySignal(data[:,1], FSAMP, 'BVP', TSTART)

ibi = ph.BeatFromBP()(bvp) #TODO: ibi should have the same indexes of original signal

'''
#EDA
FILE = '/home/andrea/Trento/DATI/FCA/NightsAndrea/18-19_Ago/CsvOutput-CSV/data_EmpaticaDevice-Empatica E4 - A0051F/GSR.csv'
data = np.array(pd.read_csv(FILE))
eda = ph.EvenlySignal(data[:,1], 4, 'EDA', 0)

eda_f = ph.DenoiseEDA(threshold = 0.2)(eda) #OK
'''