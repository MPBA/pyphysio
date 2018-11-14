# coding=utf-8
from __future__ import division

import unittest
from . import ph, TestData, np

__author__ = 'aleb'


class FiltersTest(unittest.TestCase):
    def test_general(self):
        # %%
        FSAMP = 2048
        TSTART = 0

        # %%
        ecg = ph.EvenlySignal(TestData.ecg(), sampling_freq=FSAMP, signal_nature='ECG', start_time=TSTART)

        # %%
        # TEST Normalize
        self.assertAlmostEqual(np.mean(ph.Normalize(norm_method='standard')(ecg)), 0)

        self.assertEqual(np.min(ph.Normalize(norm_method='min')(ecg)), 0)

        self.assertEqual(np.min(ph.Normalize(norm_method='maxmin')(ecg)), 0)
        self.assertEqual(np.max(ph.Normalize(norm_method='maxmin')(ecg)), 1)

        ecg_ = ph.Normalize(norm_method='custom', norm_bias=4, norm_range=0.1)(ecg)
        self.assertAlmostEqual(np.max(ecg_) - np.min(ecg_), 16.41, delta=0.005)

        # %%
        # TEST Diff
        s = ph.EvenlySignal(np.arange(1000), sampling_freq=FSAMP, start_time=TSTART)
        # degree = [5, 50, -1]
        degree = 5
        self.assertAlmostEqual(np.mean(ph.Diff(degree=degree)(s)), degree)  # OK

        # %% TEST IIRFilter
        self.assertAlmostEqual(int(np.max(ph.IIRFilter(fp=10, fs=70)(ecg)) * 10000), 8238)
        self.assertAlmostEqual(int(np.max(ph.IIRFilter(fp=70, fs=45)(ecg)) * 10000), 2144)
        self.assertAlmostEqual(int(np.max(ph.IIRFilter(fp=[5, 25], fs=[0.05, 50], ftype='ellip')(ecg)) * 10000), 8786)

        # %%

        # TEST ConvolutionalFilter
        self.assertAlmostEqual(np.max(ph.ConvolutionalFilter(irftype='gauss', win_len=0.1)(ecg)), .7501, delta=.0001)
        self.assertAlmostEqual(np.max(ph.ConvolutionalFilter(irftype='rect', win_len=0.1)(ecg)), .4022, delta=.0001)
        self.assertAlmostEqual(np.max(ph.ConvolutionalFilter(irftype='triang', win_len=0.1)(ecg)), .5000, delta=.0001)
        self.assertAlmostEqual(
            np.max(ph.ConvolutionalFilter(irftype='dgauss', win_len=0.1, normalize=False)(ecg)), .8823, delta=.0001)

        irf = np.r_[np.arange(50, 500, 5)]
        ecg_cfC = ph.ConvolutionalFilter(irftype='custom', irf=irf, normalize=True)(ecg)
        self.assertEqual(int(np.max(ecg_cfC) * 10000), 7723)

        # %%
        # TEST DeConvolutionalFilter
        ecg_df = ph.DeConvolutionalFilter(irf=irf, normalize=True, deconv_method='fft')(ecg_cfC)  # OK
        self.assertEqual(int(np.max(ecg_df) * 10000), 48973)
