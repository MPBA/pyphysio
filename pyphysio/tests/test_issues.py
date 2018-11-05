# coding=utf-8
from __future__ import print_function
from __future__ import division

from . import ph
import numpy as np

import unittest

__author__ = 'aleb'


# noinspection PyArgumentEqualDefault
class GeneralTest(unittest.TestCase):
    def test_issue24(self):
        FSAMP = 100
        n = np.arange(1000)
        t = n / FSAMP
        freq = 1

        # create reference signal
        sinusoid = ph.EvenlySignal(np.sin(2 * np.pi * freq * t), sampling_freq=FSAMP, signal_nature='', start_time=0)

        selection = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

        # remove some samples
        sinusoid_unevenly = ph.UnevenlySignal(
            np.delete(sinusoid.get_values(), selection),
            sampling_freq=FSAMP, signal_nature='', start_time=None,
            x_values=np.delete(sinusoid.get_times(), selection),
            x_type='instants')

        unev_t = sinusoid_unevenly.get_times()
        even_t = np.delete(sinusoid.get_times(), selection)
        error_sum = np.sum(sinusoid_unevenly.get_times() - np.delete(sinusoid.get_times(), selection))
        error_sum1 = np.sum(unev_t - even_t)
        where_error = np.where(sinusoid_unevenly.get_times() != np.delete(sinusoid.get_times(), selection))
        where_in_even_t = even_t[where_error]
        where_in_unev_t = unev_t[where_error]
        errors = where_in_unev_t - where_in_even_t

        # 69 errors of -0.01

        self.assertEquals(error_sum, error_sum1)
        self.assertEquals(error_sum, 0)
        self.assertEquals(error_sum, np.sum(errors))

    def test_issue22(self):
        values = np.arange(1, 11)
        instan = np.arange(1, 11)

        signal_unevenly = ph.UnevenlySignal(values=values,
                                            sampling_freq=100,
                                            signal_nature='',
                                            start_time=None,
                                            x_values=instan,
                                            x_type='instants')

        one = signal_unevenly.get_times()[0]  # = 1 <=OK
        signal_evenly = signal_unevenly.to_evenly()
        zero = signal_evenly.get_times()[0]  # = 0 <= KO

        self.assertEqual(one, 1)
        self.assertEqual(zero, 1)

        signal_unevenly = ph.UnevenlySignal(values=values,
                                            sampling_freq=100,
                                            signal_nature='',
                                            start_time=0,
                                            x_values=instan,
                                            x_type='instants')

        one = signal_unevenly.get_times()[0]  # = 1 <=OK
        signal_evenly = signal_unevenly.to_evenly()
        zero = signal_evenly.get_times()[0]  # = 0 <= KO

        self.assertEqual(one, 1)
        self.assertEqual(zero, 1)

    def test_issue18_unev(self):
        values = np.arange(1, 11)
        instan = np.arange(1, 11)
        s = ph.UnevenlySignal(values=values,
                              sampling_freq=100,
                              signal_nature='',
                              start_time=None,
                              x_values=instan,
                              x_type='instants')

        pickled = s.pickleable

        s2 = ph.Signal.from_pickleable(pickled)

        self.assertEqual(s.get_sampling_freq(), s2.get_sampling_freq())
        self.assertEqual(s.get_start_time(), s2.get_start_time())
        self.assertEqual(s.get_end_time(), s2.get_end_time())
        self.assertEqual(str(s), str(s2))

    def test_issue18_even(self):
        s = ph.EvenlySignal(values=np.sin(2 * np.pi * np.arange(1000, step=0.01)),
                            sampling_freq=100,
                            signal_nature='',
                            start_time=0)

        pickled = s.pickleable

        s2 = ph.Signal.from_pickleable(pickled)

        self.assertEqual(s.get_sampling_freq(), s2.get_sampling_freq())
        self.assertEqual(s.get_start_time(), s2.get_start_time())
        self.assertEqual(s.get_end_time(), s2.get_end_time())
        self.assertEqual(str(s), str(s2))

    @staticmethod
    def test_issue39():
        ecg = ph.EvenlySignal(ph.TestData.ecg(), sampling_freq=1024)
        algos = [
            ph.PNNx(name="PNN50", threshold=50),
            ph.PNNx(name="PNN20", threshold=20),
            ph.PNNx(name="PNN10", threshold=10),
            ph.Mean(name="Mean"),
            ph.StDev(name="Median"),
        ]

        # %%

        # create fake label
        label = np.zeros(len(ecg))
        label[int(len(ecg) / 2):] = 1
        label[int(3 * len(ecg) / 4):] = 2
        label = ph.EvenlySignal(label, sampling_freq=1024)

        # label based windowing
        label_based = ph.LabelSegments(labels=label)

        # %%

        fixed_length = ph.FixedSegments(step=5, width=20, labels=label)
        result, col_names = ph.fmap(fixed_length, algos, ecg)

        assert result is not None
        assert col_names is not None

        # %%

        # (optional) IIR filtering : remove high frequency noise
        ecg = ph.IIRFilter(fp=45, fs=50, ftype='ellip')(ecg)

        # normalization : normalize data
        ecg = ph.Normalize(norm_method='standard')(ecg)

        # resampling : increase the sampling frequency by cubic interpolation
        ecg = ecg.resample(fout=4096, kind='cubic')
        
        ibi = ph.BeatFromECG()(ecg)

        result, col_names = ph.fmap(label_based(ibi), algos)

        assert result is not None
        assert col_names is not None

    @staticmethod
    def test_issue40():
        # import data and creating a signal

        ecg_data = ph.TestData.ecg()
        fsamp = 2048
        ecg = ph.EvenlySignal(values=ecg_data, sampling_freq=fsamp, signal_nature='ecg')

        # ** Step 1: Filtering and preprocessing **

        # (optional) IIR filtering : remove high frequency noise
        ecg = ph.IIRFilter(fp=45, fs=50, ftype='ellip')(ecg)

        # normalization : normalize data
        ecg = ph.Normalize(norm_method='standard')(ecg)

        # resampling : increase the sampling frequency by cubic interpolation
        ecg = ecg.resample(fout=4096, kind='cubic')

        # ** Step 2: Information Extraction **
        ibi = ph.BeatFromECG()(ecg)

        # define a list of indicators we want to compute
        hrv_indicators = [ph.Mean(name='RRmean'), ph.StDev(name='RRstd'), ph.RMSSD(name='rmsSD'),
                          ph.PowerInBand(name='HF', interp_freq=4, freq_max=0.4, freq_min=0.15, method='ar'),
                          ph.PowerInBand(name='LF', interp_freq=4, freq_max=0.15, freq_min=0.04, method='ar')
                          ]

        # create fake label
        label = np.zeros(1200)
        label[300:600] = 1
        label[900:1200] = 2
        label = ph.EvenlySignal(label, sampling_freq=10, signal_nature='label')

        # fixed length windowing
        fixed_length = ph.FixedSegments(step=5, width=20, labels=label, drop_mixed=False, drop_shorter=True)

        indicators, col_names = ph.fmap(fixed_length(ibi), hrv_indicators)

        print(indicators[:, 0:3])

        # label based windowing
        label_based = ph.LabelSegments(labels=label)

        indicators, col_names = ph.fmap(label_based(ibi), hrv_indicators)

        print(indicators[:, 0:3])

        # custom based windowing
        custom_based = ph.CustomSegments(begins=[0, 30, 60, 90], ends=[30, 60, 90, label.get_duration()], labels=label)

        indicators, col_names = ph.fmap(custom_based, hrv_indicators, ibi)

        print(indicators[:, 0:3])

    @staticmethod
    def test_issue40_10():
        ecg_data = ph.TestData.ecg()
        fsamp = 2048
        ecg = ph.EvenlySignal(values=ecg_data, sampling_freq=fsamp, signal_nature='ecg')

        # resampling : increase the sampling frequency by cubic interpolation
        ecg = ecg.resample(fout=4096, kind='cubic')

        ibi = ph.BeatFromECG()(ecg)

        # create fake label
        label = np.zeros(1200)
        label[300:600] = 1
        label[900:1200] = 2

        label = ph.EvenlySignal(label, sampling_freq=10)

        # label based windowing
        label_based = ph.LabelSegments(labels=label)

        assert len([x for x in label_based(ibi)]) == 4

    @staticmethod
    def test_issue48():
        ecg = ph.EvenlySignal(values=ph.TestData.ecg(), sampling_freq=2048, signal_nature='ecg')
        ibi = ph.BeatFromECG()(ecg)

        # create fake label
        label = np.zeros(1200)
        label[300:600] = 1
        label[900:1200] = 2
        label = ph.EvenlySignal(label, sampling_freq=10, signal_nature='label')

        t_start = [0.5, 15, 88.7]
        t_stop = [5, 21, 110.4]
        custom_segments = ph.CustomSegments(begins=t_start, ends=t_stop, labels=label)

        indicators, col_names = ph.fmap(custom_segments, [ph.Mean()], ibi)

        assert len(indicators) == 2

        assert indicators[0][0] == t_start[0]
        assert indicators[1][0] == t_start[1]

        assert indicators[0][1] == t_stop[0]
        assert indicators[1][1] == t_stop[1]

        assert indicators[0][2] == 0
        assert indicators[1][2] == 0
