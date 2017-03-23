# coding=utf-8
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

    def test_issue18_unev(self):
        values = np.arange(1, 11)
        instan = np.arange(1, 11)
        s = ph.UnevenlySignal(values=values,
                              sampling_freq=100,
                              signal_nature='',
                              start_time=None,
                              x_values=instan,
                              x_type='instants')

        pickled = s.p

        s2 = ph.UnevenlySignal.unp(pickled)

        self.assertEqual(s.get_sampling_freq(), s2.get_sampling_freq())
        self.assertEqual(s.get_start_time(), s2.get_start_time())
        self.assertEqual(s.get_end_time(), s2.get_end_time())
        self.assertEqual(str(s), str(s2))

    def test_issue18_even(self):
        s = ph.EvenlySignal(values=np.sin(2 * np.pi * np.arange(1000, step=0.01)),
                            sampling_freq=100,
                            signal_nature='',
                            start_time=0)

        pickled = s.p

        s2 = ph.EvenlySignal.unp(pickled)

        self.assertEqual(s.get_sampling_freq(), s2.get_sampling_freq())
        self.assertEqual(s.get_start_time(), s2.get_start_time())
        self.assertEqual(s.get_end_time(), s2.get_end_time())
        self.assertEqual(str(s), str(s2))

    def test_issue39(self):
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

        # %%

        # (optional) IIR filtering : remove high frequency noise
        ecg = ph.IIRFilter(fp=45, fs=50, ftype='ellip')(ecg)

        # normalization : normalize data
        ecg = ph.Normalize(norm_method='standard')(ecg)

        # resampling : increase the sampling frequency by cubic interpolation
        ecg = ecg.resample(fout=4096, kind='cubic')
        fsamp = 4096

        ibi = ph.BeatFromECG()(ecg)

        result, col_names = ph.fmap(label_based(ibi), algos)
