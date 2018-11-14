# coding=utf-8
from __future__ import division
from . import *


def test_ibi_extraction():
    # %%
    FSAMP = 2048
    TSTART = 0

    # %%
    # TEST IBI EXTRACTION FROM ECG
    ecg = ph.EvenlySignal(TestData.ecg(), sampling_freq=FSAMP, signal_nature='ECG', start_time=TSTART)

    ecg = ecg.resample(fout=4096, kind='cubic')

    ibi = ph.BeatFromECG()(ecg)
    assert (int(np.mean(ibi) * 10000) == 8619)

    # %% Test ibi correction
    id_bad_ibi = ph.BeatOutliers(cache=3, sensitivity=0.25)(ibi)
    assert (len(id_bad_ibi) == 0)

    def insert_ibi(id_ibi, ibi_good):
        t_ibi_bad = ibi_good.get_times()
        v_ibi_bad = ibi_good.get_values()
        t_ibi_bad_left = t_ibi_bad[id_ibi]
        t_ibi_bad_right = t_ibi_bad[id_ibi + 1]
        t_ibi_add = t_ibi_bad_left + (t_ibi_bad_right - t_ibi_bad_left) / 4

        t_ibi_bad = np.r_[t_ibi_bad[:id_ibi + 1], t_ibi_add, t_ibi_bad[id_ibi + 1:]]

        v_ibi_add = t_ibi_add - t_ibi_bad_left
        v_ibi_bad[id_ibi + 1] = t_ibi_bad_right - t_ibi_add
        v_ibi_bad = np.r_[v_ibi_bad[:id_ibi + 1], v_ibi_add, v_ibi_bad[id_ibi + 1:]]

        return (ph.UnevenlySignal(v_ibi_bad, sampling_freq=ibi_good.get_sampling_freq(),
                                  start_time=ibi_good.get_start_time(),
                                  signal_nature=ibi_good.get_signal_nature(),
                                  x_values=t_ibi_bad, x_type='instants'))

    ibi_bad = insert_ibi(20, ibi.copy())
    id_bad_ibi = ph.BeatOutliers(cache=3, sensitivity=0.25)(ibi_bad)

    assert (id_bad_ibi[0] == 21 and id_bad_ibi[1] == 22)

    # %%
    # TEST IBI EXTRACTION FROM BVP
    bvp = ph.EvenlySignal(TestData.bvp(), sampling_freq=FSAMP, signal_nature='BVP', start_time=TSTART)
    bvp = bvp.resample(fout=4096, kind='cubic')

    ibi = ph.BeatFromBP()(bvp)

    id_bad_ibi = ph.BeatOutliers(cache=3, sensitivity=0.25)(ibi)
    assert (np.sum(id_bad_ibi) == 744)
