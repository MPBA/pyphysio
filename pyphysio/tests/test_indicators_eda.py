# coding=utf-8
from __future__ import division

from . import ph, TestData, np, approx


def test_indicators_eda():
    # %%
    # EDA
    FSAMP = 2048

    eda = ph.EvenlySignal(TestData.eda(), sampling_freq=FSAMP, signal_nature='EDA', start_time=0)

    # pre-processing
    # downsampling
    eda = eda.resample(8)

    # filter
    eda = ph.IIRFilter(fp=0.8, fs=1.1)(eda)

    # estimate driver
    driver = ph.DriverEstim(t1=0.75, t2=2)(eda)
    phasic, tonic, driver_no_peak = ph.PhasicEstim(delta=0.1)(driver)

    # %%
    assert ph.Mean()(phasic) == approx(.0568, abs=0.00005)

    mx = ph.Max()(phasic)
    pks_max = ph.PeaksMax(delta=0.1)(phasic)

    assert (pks_max == mx)

    idx_mx, idx_mn, mx, mn = ph.PeakDetection(delta=0.1)(phasic)
    st, sp = ph.PeakSelection(indices=idx_mx, win_pre=2, win_post=2)(phasic)

    n_peaks = ph.PeaksNum(delta=0.1, win_pre=2, win_post=2)(phasic)

    assert (n_peaks == 24)
    assert (np.sum(st) == 12501)
    assert (np.sum(sp) == 13031)

    # %%
    # FAKE PHASIC
    data = np.zeros(80)
    phasic = ph.EvenlySignal(data, sampling_freq=8, signal_nature='PHA', start_time=0)

    phasic[36] = -0.05
    phasic[37] = 0.12
    phasic[39] = 0.10
    phasic[40] = 0.2
    phasic[41] = 0.10
    phasic[43] = 0.12
    phasic[44] = -0.05

    pks_max = ph.PeaksMax(delta=0.1)(phasic)
    assert (pks_max == 0.2)

    pks_min = ph.PeaksMin(delta=0.1)(phasic)
    assert (pks_min == 0.12)

    pks_mean = ph.PeaksMean(delta=0.1)(phasic)
    assert (pks_mean == np.mean([0.12, 0.2, 0.12]))

    dur_max = ph.DurationMax(delta=0.1, win_pre=2, win_post=2)(phasic)
    assert (dur_max == 0.5)

    dur_min = ph.DurationMin(delta=0.1, win_pre=2, win_post=2)(phasic)
    assert (dur_min == 0.25)

    dur_mean = ph.DurationMean(delta=0.1, win_pre=2, win_post=2)(phasic)
    assert (dur_mean == np.mean([0.25, 0.25, 0.5]))
