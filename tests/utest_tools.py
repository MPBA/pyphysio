from __future__ import division
import numpy as np
import pandas as pd

from context import ph, Asset

# BVP
FILE = Asset.BVP
FSAMP = 64
TSTART = 0

data = np.array(pd.read_csv(FILE))
bvp = ph.EvenlySignal(data[:, 1], FSAMP, 'BVP', TSTART)

# EDA
FILE = Asset.GSR
data = np.array(pd.read_csv(FILE))
eda = ph.EvenlySignal(data[:, 1], 4, 'EDA', 0)

eda = ph.DenoiseEDA(threshold=0.2)(eda)  # OK
eda = ph.ConvolutionalFilter(irftype='gauss', normalize=True, win_len=2)(eda)
# TEST SignalRange
rng_bvp = ph.SignalRange(win_len=2, win_step=2, smooth=False)(bvp)  # OK

# TEST PeakDetection
mx, mn, _, _ = ph.PeakDetection(deltas=rng_bvp * 0.7)(bvp)

mx2, mn2, _, _ = ph.PeakDetection(delta=0.02)(eda)

# TEST PeakSelection
st, sp = ph.PeakSelection(maxs=mx2, pre_max=3, post_max=5)(eda)

# TEST PSD
FSAMP = 100
n = np.arange(1000)
t = n / FSAMP
freq = 2.5
sinusoid = ph.EvenlySignal(np.sin(2 * np.pi * freq * t), FSAMP, '', 0)

f, psd = ph.PSD(psd_method='ar', nfft=1024, window='hanning')(sinusoid)  # OK

# TEST Energy
energy = ph.Energy(win_len=5, win_step=2.5, smooth=True)(bvp)  # OK

# TEST Maxima
mx = ph.Maxima(method='complete', win_len=2, win_step=1)(eda)  # TODO: che senso ha il windowing? /// Was?

# TEST Minima
mn = ph.Minima(method='windowing', win_len=20, win_step=5)(eda)  # TODO: che senso ha il windowing?

# TEST CreateTemplate() #TODO: test
# ...

# TEST BootstrapEstimation
bt_mean = ph.BootstrapEstimation(func=np.mean, N=100, k=0.5)(eda)  # OK
