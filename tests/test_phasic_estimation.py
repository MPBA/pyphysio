# coding=utf-8
from __future__ import division
from context import ph, np, Assets, approx


def test_phasic_estimation():
    # %%
    FSAMP = 2048

    # %%
    # EDA
    eda = ph.EvenlySignal(Assets.eda(), sampling_freq=FSAMP, signal_nature='EDA', start_time=0)

    # preprocessing
    # downsampling
    eda_r = eda.resample(4)

    # filter
    eda = ph.IIRFilter(fp=0.8, fs=1.1)(eda_r)

    # %%
    # estimate driver
    driver = ph.DriverEstim(T1=0.75, T2=2)(eda)
    assert (int(np.mean(driver) * 10000) == 18255)
    assert (isinstance(driver, ph.EvenlySignal))

    # %%
    phasic, tonic, driver_no_peak = ph.PhasicEstim(delta=0.1)(driver)
    assert np.mean(phasic) == approx(.0485)
    assert (isinstance(phasic, ph.EvenlySignal))
