from __future__ import division
import numpy as np
import pandas as pd

import pyPhysio as ph

FILE = '/home/andrea/Trento/DATI/SYNC/F/F18.txt'

data = np.array(pd.read_csv(FILE, skiprows=8, header=None))
id_eda1 = 1
id_eda2 = 3
id_emg1 = 4
id_emg2 = 7
id_ecg1 = 5
id_ecg2 = 6
id_tt = 2

ecg1 = data[:, id_ecg1]

FSAMP = 2048
TSTART = 0

ecg = ph.EvenlySignal(ecg1, FSAMP, 'ECG', TSTART) # OK
ecg.plot() # OK

# TEST Normalize
ecg_n1 = ph.Normalize(norm_method='mean')(ecg) # OK
ecg_n2 = ph.Normalize(norm_method='standard')(ecg) # TODO: non funziona
ecg_n3 = ph.Normalize(norm_method='min')(ecg) # OK
ecg_n4 = ph.Normalize(norm_method='maxmin')(ecg) # OK
ecg_n5 = ph.Normalize(norm_method='custom', norm_bias = 4, norm_range = 0.1)(ecg) # OK

# TEST Diff
s = ph.EvenlySignal(np.arange(1000), FSAMP, '', TSTART)
ph.Diff(degree=1)(s).plot() # OK

# TEST IIRFilter (default parameters)
ecg_lp = ph.IIRFilter(fp = 45, fs = 70)(ecg) # OK
ecg_hp = ph.IIRFilter(fp = 70, fs = 45)(ecg) # OK
ecg_bp = ph.IIRFilter(fp = [70, 100], fs = [45, 150])(ecg) # OK
ecg_notch50 = ph.IIRFilter(fp = [45, 55], fs = [50, 50.1])(ecg) # OK

# TEST MatchedFilter
template = ecg[2000:3000]

ecg_m = ph.MatchedFilter(template=np.array(template))(ecg) #TODO: errore strano con cache

# TEST ConvolutionalFilter
ecg_cf1 = ph.ConvolutionalFilter(irftype = 'gauss', win_len = 0.1)(ecg) #OK
ecg_cf2 = ph.ConvolutionalFilter(irftype = 'rect', win_len = 0.1)(ecg) #OK
ecg_cf3 = ph.ConvolutionalFilter(irftype = 'triang', win_len = 0.1)(ecg) #OK
ecg_cf4 = ph.ConvolutionalFilter(irftype = 'dgauss', win_len = 0.1)(ecg) #TODO: Check results
ecg_cf5 = ph.ConvolutionalFilter(irftype = 'custom', irf = [0, 1,2,1,0], normalize=False)(ecg) #OK

# TEST DeConvolutionalFilter (VERY SLOW!!!)
ecg_df1 = ph.DeConvolutionalFilter(irf = [0, 0, 2, 0, 0], normalize = True)(ecg) #OK

# TEST DenoiseEDA
FILE = '/home/andrea/Trento/DATI/FCA/NightsAndrea/18-19_Ago/CsvOutput-CSV/data_EmpaticaDevice-Empatica E4 - A0051F/GSR.csv'
data = np.array(pd.read_csv(FILE))
eda = ph.EvenlySignal(data[:,1], 4, 'EDA', 0)

eda_f = ph.DenoiseEDA(threshold = 0.2)(eda) #OK