# coding=utf-8
from __future__ import division

from . import ph, np, TestData, approx


# %%
def test_tools():
    # %%
    FSAMP = 2048
    TSTART = 0

    # %%
    bvp = ph.EvenlySignal(TestData.bvp(), sampling_freq=FSAMP, signal_nature='BVP', start_time=TSTART)

    # %%
    # TEST SignalRange
    rng_bvp = ph.SignalRange(win_len=1, win_step=0.5, smooth=True)(bvp)  # OK
    assert (int(np.max(rng_bvp) * 100) == 1746)

    # %%
    # TEST PeakDetection
    idx_mx, idx_mn, mx, mn = ph.PeakDetection(delta=rng_bvp * 0.5)(bvp)

    assert (np.sum(idx_mx) == 16818171)
    assert (np.sum(idx_mn) == 16726017)

    # %%
    # EDA
    eda = ph.EvenlySignal(TestData.eda(), sampling_freq=FSAMP, signal_nature='EDA', start_time=TSTART)
    eda = eda.resample(fout=8)

    eda = eda.resample(8)

    # filter
    eda = ph.IIRFilter(fp=0.8, fs=1.1)(eda)

    driver = ph.DriverEstim()(eda)
    phasic, _, __ = ph.PhasicEstim(delta=0.02)(driver)

    idx_mx, idx_mn, mx, mn = ph.PeakDetection(delta=0.02)(phasic)

    # TEST PeakSelection
    st, sp = ph.PeakSelection(indices=idx_mx, win_pre=2, win_post=2)(phasic)
    assert (np.sum(st) == 22985)

    # %%
    # TEST PSD
    FSAMP = 100
    n = np.arange(1000)
    t = n / FSAMP
    freq = 2.5

    sinusoid = ph.EvenlySignal(np.sin(2 * np.pi * freq * t), sampling_freq=FSAMP, signal_nature='', start_time=0)

    f, psd = ph.PSD(method='welch', nfft=4096, window='hanning')(sinusoid)

    assert (f[np.argmax(psd)] == approx(2.5, abs=0.02))

    sinusoid_unevenly = ph.UnevenlySignal(np.delete(sinusoid.get_values(), np.arange(10, 200)),
                                          sampling_freq=FSAMP,
                                          start_time=0,
                                          x_values=np.delete(sinusoid.get_times(), np.arange(10, 200)),
                                          x_type='instants')

    f, psd = ph.PSD(method='welch', nfft=4096, window='hanning')(sinusoid_unevenly)

    assert (f[np.argmax(psd)] == approx(2.5, abs=0.02))

    # %%
    # TEST Maxima
    idx_mx, mx = ph.Maxima(method='windowing', win_len=1, win_step=0.5)(bvp)
    assert (np.sum(idx_mx) == 16923339)

    # TEST Minima
    idx_mn, mn = ph.Minima(method='windowing', win_len=1, win_step=0.5)(bvp)
    assert (np.sum(idx_mn) == 17939276)
