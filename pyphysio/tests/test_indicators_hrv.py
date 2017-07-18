# coding=utf-8
from __future__ import division
from . import ph, np, TestData, approx


def test_indicators():
    # %%
    FSAMP = 2048
    TSTART = 0

    # %%
    # TEST IBI EXTRACTION FROM ECG
    ecg = ph.EvenlySignal(TestData.ecg(), sampling_freq=FSAMP, signal_nature='ECG', start_time=TSTART)

    ecg = ecg.resample(fout=4096, kind='cubic')

    ibi = ph.BeatFromECG()(ecg)
    # %%
    # TEST Time domain
    assert ph.Mean()(ibi) == approx(.86195, abs=.00005)
    assert ph.StDev()(ibi) == approx(.06027, abs=.00005)
    assert ph.Median()(ibi) == approx(.87133, abs=.00005)
    assert ph.Range()(ibi) == approx(.25488, abs=.00005)
    assert ph.RMSSD()(ibi) == approx(.03260, abs=.00005)
    assert ph.SDSD()(ibi) == approx(.03260, abs=.00005)

    # TEST Frequency domain
    assert ph.PowerInBand(method='welch', interp_freq=4, freq_max=0.04, freq_min=0.00001)(ibi) == approx(127.1227,
                                                                                                         abs=.00005)
    assert ph.PowerInBand(method='welch', interp_freq=4, freq_max=0.15, freq_min=0.04)(ibi) == approx(259.99196,
                                                                                                      abs=.00005)
    assert ph.PowerInBand(method='welch', interp_freq=4, freq_max=0.4, freq_min=0.15)(ibi) == approx(120.18714,
                                                                                                     abs=.00005)

    assert ph.PNNx(threshold=10)(ibi) == approx(.3453, abs=.00005)
    assert ph.PNNx(threshold=25)(ibi) == approx(.2158, abs=.00005)
    assert ph.PNNx(threshold=50)(ibi) == approx(.04316, abs=.00005)

    # %%
    # Test with FAKE IBI
    idx_ibi = np.arange(0, 101, 10).astype(float)
    ibi = ph.UnevenlySignal(np.diff(idx_ibi), x_values=idx_ibi[1:], sampling_freq=10, signal_nature='IBI',
                            x_type='indices')

    assert int(ph.Mean()(ibi)) == 10
    assert int(ph.StDev()(ibi)) == 0

    assert int(ph.Median()(ibi)) == 10
    assert int(ph.Range()(ibi)) == 0

    assert int(ph.RMSSD()(ibi)) == 0
    assert int(ph.SDSD()(ibi)) == 0

    assert ph.AUC()(ibi) == 10

    # TEST Non linear
    assert ph.PoincareSD1()(ibi) == 0
    assert ph.PoincareSD2()(ibi) == 0
    assert ph.ApproxEntropy()(ibi) == 0
    assert ph.SampleEntropy()(ibi) == 0

    ibi[-1] = 10.011

    assert ph.Mean()(ibi) != 10
    assert ph.StDev()(ibi) != 0

    assert int(ph.Median()(ibi)) == 10

    assert ph.Range()(ibi) != 0

    assert ph.RMSSD()(ibi) != 0
    assert ph.SDSD()(ibi) != 0

    assert ph.PNNx(threshold=10)(ibi) != 0
    assert ph.PNNx(threshold=25)(ibi) == 0
    assert ph.PNNx(threshold=50)(ibi) == 0

    assert ph.PoincareSD1SD2()(ibi) != 1
