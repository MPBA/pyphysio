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
    assert int(ph.Mean()(ibi) * 10000) == 8619
    assert int(ph.StDev()(ibi) * 10000) == 602

    assert int(ph.Median()(ibi) * 10000) == 8679
    assert int(ph.Range()(ibi) * 10000) == 2548

    assert int(ph.RMSSD()(ibi) * 10000) == 328
    assert int(ph.SDSD()(ibi) * 10000) == 328

    # TEST Frequency domain
    # TODO: almost equal below
    assert int(ph.PowerInBand(interp_freq=4, method='welch', freq_max=0.04, freq_min=0.00001)(ibi) * 10000) == 1271328
    assert int(ph.PowerInBand(interp_freq=4, method='welch', freq_max=0.15, freq_min=0.04)(ibi) * 10000) == 2599850
    assert int(ph.PowerInBand(method='welch', interp_freq=4, freq_max=0.4, freq_min=0.15)(ibi) * 10000) == 1201839

    assert int(ph.PNNx(threshold=10)(ibi) * 10000) == 3453
    assert int(ph.PNNx(threshold=25)(ibi) * 10000) == 2158
    assert int(ph.PNNx(threshold=50)(ibi) * 10000) == 431

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

    # TODO: almost equal belo
    assert ph.Range()(ibi) == approx(0.011)

    assert ph.RMSSD()(ibi) != 0
    assert ph.SDSD()(ibi) != 0

    assert ph.PNNx(threshold=10)(ibi) == 0.1
    assert ph.PNNx(threshold=25)(ibi) == 0
    assert ph.PNNx(threshold=50)(ibi) == 0

    # TODO almost equal
    assert ph.PoincareSD1SD2()(ibi) == approx(1)
