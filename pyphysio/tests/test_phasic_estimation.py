# coding=utf-8
from __future__ import division
from . import ph, np, TestData, approx


def test_phasic_estimation():
    # %%
    FSAMP = 2048

    # %%
    # EDA
    eda = ph.EvenlySignal(TestData.eda(), sampling_freq=FSAMP, signal_nature='EDA', start_time=0)

    # preprocessing
    # downsampling
    eda = eda.resample(8)

    # filter
    eda = ph.IIRFilter(fp=0.8, fs=1.1)(eda)

    # %%
    # estimate driver
    driver = ph.DriverEstim(t1=0.75, t2=2.5)(eda)
    assert (int(np.mean(driver) * 10000) == 18286)
    assert (isinstance(driver, ph.EvenlySignal))

    # %%
    phasic, tonic, driver_no_peak = ph.PhasicEstim(delta=0.1)(driver)
    assert np.mean(phasic) == approx(.0656, abs=0.0005)
    assert (isinstance(phasic, ph.EvenlySignal))
