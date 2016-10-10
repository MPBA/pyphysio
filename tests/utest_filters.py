from __future__ import division
import numpy as np
import pandas as pd

from context import ph, Asset

FILE = Asset.F18

ecg1 = np.array(pd.read_csv(FILE))

FSAMP = 2048
TSTART = 0

ecg = ph.EvenlySignal(ecg1, FSAMP, 'ECG', TSTART)  # OK
ecg.plot()  # OK

# TEST Normalize
ecg_n1 = ph.Normalize(norm_method='mean')(ecg)  # OK
ecg_n2 = ph.Normalize(norm_method='standard')(ecg)  # OK
ecg_n3 = ph.Normalize(norm_method='min')(ecg)  # OK
ecg_n4 = ph.Normalize(norm_method='maxmin')(ecg)  # OK
ecg_n5 = ph.Normalize(norm_method='custom', norm_bias=4, norm_range=0.1)(ecg)  # OK

# TEST Diff
s = ph.EvenlySignal(np.arange(1000), FSAMP, '', TSTART)
ph.Diff(degree=5)(s).plot()  # OK

# TEST IIRFilter (default parameters)
ecg_lp = ph.IIRFilter(fp=100, fs=150)(ecg)  # TOCHECK
ecg_hp = ph.IIRFilter(fp=70, fs=45)(ecg)  # OK
ecg_bp = ph.IIRFilter(fp=[70, 100], fs=[45, 150])(ecg)  # OK
ecg_notch50 = ph.IIRFilter(fp=[45, 55], fs=[50, 50.1])(ecg)  # OK

# TEST MatchedFilter
template = ecg[2000:3000]

ecg_m = ph.MatchedFilter(template=template.get_values())(ecg) 

# TEST ConvolutionalFilter
# TODO (Andrea): error, win_len=0.1 is too few. See ConvolutionalFilter TODOs
ecg_cf1 = ph.ConvolutionalFilter(irftype='gauss', win_len=0.1)(ecg)  # OK
ecg_cf2 = ph.ConvolutionalFilter(irftype='rect', win_len=0.1)(ecg)  # OK
ecg_cf3 = ph.ConvolutionalFilter(irftype='triang', win_len=0.1)(ecg)  # OK
ecg_cf4 = ph.ConvolutionalFilter(irftype='dgauss', win_len=0.1)(ecg)  # TODO: Check results
ecg_cf5 = ph.ConvolutionalFilter(irftype='custom', irf=[0, 1, 2, 1, 0], normalize=True)(ecg)  # OK

# TEST DeConvolutionalFilter (VERY SLOW!!!)
ecg_df1 = ph.DeConvolutionalFilter(irf=[0, 0, 2, 0, 0], normalize=True)(ecg)  # OK

# TEST DenoiseEDA
FILE = Asset.GSR
data = np.array(pd.read_csv(FILE))
eda = ph.EvenlySignal(data[:, 1], 4, 'EDA', 0)

eda_f = ph.DenoiseEDA(threshold=0.2)(eda)  # OK
