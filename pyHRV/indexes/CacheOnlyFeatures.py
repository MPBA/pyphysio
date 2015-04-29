from __future__ import division

# coding=utf-8
import spectrum

__author__ = 'AleB'

import numpy as np
from scipy import signal

from pyHRV.Utility import ordered_subsets, interpolate_ibi
from pyHRV.indexes.BaseFeatures import CacheOnlyFeature
from pyHRV.PyHRVSettings import MainSettings as Sett


class FFTCalc(CacheOnlyFeature):
    @classmethod
    def _compute(cls, data, params):
        """
        Calculates the FFT data to cache
        @param data: DataSeries object
        @type data: DataSeries
        @param params: Dict containing interp_freq: Frequency for the interpolation before the pow. spec. estimation.
        @return: Data to cache: (bands, powers)
        @rtype: (array, array)
        """
        if 'interp_freq' not in params or params['interp_freq'] is None:
            params['interp_freq'] = Sett.default_interpolation_freq
        rr_interp, ignored = interpolate_ibi(data.series, params['interp_freq'])  # TODO 2 Andrea: change interp. type
        interp_freq = params['interp_freq']
        hw = np.hamming(len(rr_interp))

        frame = rr_interp * hw
        frame = frame - np.mean(frame)

        spec_tmp = np.absolute(np.fft.fft(frame)) ** 2  # FFT
        powers = spec_tmp[0:(np.ceil(len(spec_tmp) / 2))]  # Only positive half of spectrum
        bands = np.linspace(start=0, stop=interp_freq / 2, num=len(powers), endpoint=True)  # frequencies vector
        return bands, powers

    @staticmethod
    def get_used_params():
        return ['interp_freq']


class PSDLombscargleCalc(CacheOnlyFeature):
    @classmethod
    def _compute(cls, data, params):
        """
        Calculates the PSD data to cache using the Lombscargle algorithm
        @param data: DataSeries object
        @type data: DataSeries
        @param params: Dict containing interp_freq Frequency for the interpolation before the pow. spec. estimation.
        @return: Data to cache: (bands, powers, total_power)
        @rtype: (array, array, float)
        """
        if 'lombscargle_stop' not in params or params['lombscargle_stop'] is None:
            params['lombscargle_stop'] = Sett.default_interpolation_freq
        if Sett.remove_mean:
            data = data - np.mean(data)
        t = np.cumsum(data)

        # TODO 5 Andrea: is it an interpolation frequency?
        # stop : scalar
        #     The end value of the sequence, unless endpoint is set to False. In that case, the sequence consists of
        #     all but the last of num + 1 evenly spaced samples, so that stop is excluded. Note that the step size
        #     changes when endpoint is False.

        bands = np.linspace(start=0, stop=params['lombscargle_stop'] / 2, num=max(128, len(data)), endpoint=True)
        bands = bands[1:]
        powers = np.sqrt(4 * (signal.lombscargle(t, data, bands) / len(data)))

        return bands, powers / np.max(powers), sum(powers) / len(powers)

    @staticmethod
    def get_used_params():
        return ['lombscargle_stop']


class PSDFFTCalc(CacheOnlyFeature):
    @classmethod
    def _compute(cls, data, params):
        """
        Calculates the PSD data to cache using the fft algorithm
        @param data: DataSeries object
        @type data: DataSeries
        @param params: Dict containing interp_freq Frequency for the interpolation before the pow. spec. estimation.
        @return: Data to cache: (bands, powers, total_power)
        @rtype: (array, array, float)
        """
        if 'interp_freq' not in params or params['interp_freq'] is None:
            params['interp_freq'] = Sett.default_interpolation_freq
        data_interp, t_interp = interpolate_ibi(data, params['interp_freq'])  # TODO 6: change interp. type
        if Sett.remove_mean:
            data_interp = data_interp - np.mean(data_interp)

        hw = np.hamming(len(data_interp))
        frame = data_interp * hw
        spec_tmp = np.absolute(np.fft.fft(frame)) ** 2  # FFT
        powers = spec_tmp[0:(np.ceil(len(spec_tmp) / 2))]

        bands = np.linspace(start=0, stop=params['interp_freq'] / 2, num=len(powers), endpoint=True)

        return bands, powers / np.max(powers), sum(powers) / len(powers)

    @staticmethod
    def get_used_params():
        return ['interp_freq']


class PSDWelchLinspaceCalc(CacheOnlyFeature):
    @classmethod
    def _compute(cls, data, params):
        """
        Calculates the PSD data to cache using the welch algorithm, uses linspace bands distribution
        @param data: DataSeries object
        @type data: DataSeries
        @param params: Dict containing interp_freq Frequency for the interpolation before the pow. spec. estimation.
        @return: Data to cache: (bands, powers, total_power)
        @rtype: (array, array, float)
        """
        if 'interp_freq' not in params or params['interp_freq'] is None:
            params['interp_freq'] = Sett.default_interpolation_freq
        data_interp, t_interp = interpolate_ibi(data, params['interp_freq'])  # TODO 6: change interp. type
        if Sett.remove_mean:
            data_interp = data_interp - np.mean(data_interp)
        bands_w, powers = signal.welch(data_interp, params['interp_freq'], nfft=max(128, len(data_interp)))
        bands = np.linspace(start=0, stop=params['interp_freq'] / 2, num=len(powers), endpoint=True)
        return bands, powers / np.max(powers), sum(powers) / len(powers)

    @staticmethod
    def get_used_params():
        return ['interp_freq']


class PSDWelchLibCalc(CacheOnlyFeature):
    @classmethod
    def _compute(cls, data, params):
        """
        Calculates the PSDWelch data to cache, uses algorithms bands distribution
        @param data: DataSeries object
        @type data: DataSeries
        @param params: Dict containing interp_freq Frequency for the interpolation before the pow. spec. estimation.
        @return: Data to cache: (bands, powers, total_power)
        @rtype: (array, array, float)
        """
        if 'interp_freq' not in params or params['interp_freq'] is None:
            params['interp_freq'] = Sett.default_interpolation_freq
        rr_interp, bt_interp = interpolate_ibi(data, params['interp_freq'])  # TODO 6: change interp. type
        bands, powers = signal.welch(rr_interp, params['interp_freq'], nfft=max(128, len(rr_interp)))
        powers = np.sqrt(powers)
        return bands, powers / np.max(powers), sum(powers) / len(powers)

    @staticmethod
    def get_used_params():
        return ['interp_freq']


class PSDAr1Calc(CacheOnlyFeature):
    @classmethod
    def _compute(cls, data, params):
        """
        Calculates the PSD data to cache using the ar_1 algorithm
        @param data: DataSeries object
        @type data: DataSeries
        @param params: Dict containing interp_freq Frequency for the interpolation before the pow. spec. estimation.
        @return: Data to cache: (bands, powers, total_power)
        @rtype: (array, array, float)
        """
        if 'interp_freq' not in params or params['interp_freq'] is None:
            params['interp_freq'] = Sett.default_interpolation_freq
        data_interp, t_interp = interpolate_ibi(data, params['interp_freq'])  # TODO 6: change interp. type
        if Sett.remove_mean:
            data_interp = data_interp - np.mean(data_interp)

        p = spectrum.Periodogram(data_interp, sampling=params['interp_freq'], NFFT=max(128, len(data_interp)))
        p()
        powers = p.get_converted_psd('onesided')
        bands = np.linspace(start=0, stop=params['interp_freq'] / 2, num=len(powers), endpoint=True)

        return bands, powers / np.max(powers), sum(powers) / len(powers)

    @staticmethod
    def get_used_params():
        return ['interp_freq']


class PSDAr2Calc(CacheOnlyFeature):
    @classmethod
    def _compute(cls, data, params):
        """
        Calculates the PSD data to cache using the ar_2 algorithm
        @param data: DataSeries object
        @type data: DataSeries
        @param params: Dict containing interp_freq Frequency for the interpolation before the pow. spec. estimation.
        @return: Data to cache: (bands, powers, total_power)
        @rtype: (array, array, float)
        """
        if 'interp_freq' not in params or params['interp_freq'] is None:
            params['interp_freq'] = Sett.default_interpolation_freq
        powers = []

        data_interp, t_interp = interpolate_ibi(data, params['interp_freq'])  # TODO 6: change interp. type
        if Sett.remove_mean:
            data_interp = data_interp - np.mean(data_interp)

        orders = range(1, Sett.ar_2_max_order + 1)
        for order in orders:
            try:
                ar, p, k = spectrum.aryule(data_interp, order=order, norm='biased')
            except AssertionError:
                ar = 1
                print("Error in ar_2 psd ayrule, assumed ar=1")
            powers = spectrum.arma2psd(ar, NFFT=max(128, len(data_interp)))
            powers = powers[0: np.ceil(len(powers) / 2)]
        else:
            print("Error in ar_2 psd, orders=0, empty powers")

        bands = np.linspace(start=0, stop=params['interp_freq'] / 2, num=len(powers), endpoint=True)

        return bands, powers / np.max(powers), sum(powers) / len(powers)

    @staticmethod
    def get_used_params():
        return ['interp_freq']


class Histogram(CacheOnlyFeature):
    @classmethod
    def _compute(cls, data, params):
        """
        Calculates the Histogram data to cache
        @param data: DataSeries object
        @type data: DataSeries
        @param params: Dict containing histogram_bins
        @return: Data to cache: (hist, bin_edges)
        @rtype: (array, array)
        """
        if 'histogram_bins' not in params or params['histogram_bins'] is None:
            params['histogram_bins'] = Sett.cache_histogram_bins
        return np.histogram(data, params['histogram_bins'])

    @staticmethod
    def get_used_params():
        return ['histogram_bins']


class HistogramMax(CacheOnlyFeature):
    @classmethod
    def _compute(cls, data, params):
        """
        Calculates the Histogram's max value
        @param data: DataSeries object
        @type data: DataSeries
        @param params: Dict containing histogram_bins
        @return: Data to cache: (hist, bin_edges)
        @rtype: (array, array)
        """
        h, b = Histogram.get(data, params['histogram_bins'])
        return np.max(h)  # TODO 2 Andrea: max h or b(max h)??

    @staticmethod
    def get_used_params():
        return ['histogram_bins']


class Diff(CacheOnlyFeature):
    @classmethod
    def _compute(cls, data, params):
        """
        Calculates the differences between consecutive values
        @param data: DataSeries object
        @type data: DataSeries
        @param params: Unused
        @return: Data to cache: diff
        @rtype: array
        """
        return np.diff(np.array(data))

    @staticmethod
    def get_used_params():
        return []


class StandardDeviation(CacheOnlyFeature):
    @classmethod
    def _compute(cls, data, params):
        """
        Calculates the standard deviation data
        @param data: DataSeries object
        @type data: DataSeries
        @param params: Unused
        @return: Data to cache: st. dev.
        @rtype: array
        """
        return np.std(np.array(data))

    @staticmethod
    def get_used_params():
        return []


class OrderedSubsets(CacheOnlyFeature):
    @classmethod
    def _compute(cls, data, params):
        """
        Calculates the the vector of the sequences of length 2 of the data
        @param data: DataSeries object
        @type data: DataSeries
        @param params: Dict containing subset_size
        @return: Data to cache: ndarray with shape (l - n + 1, n) having l=len(data) and n=subset_size
        @rtype: array
        """
        return ordered_subsets(data, params['subset_size'])

    @staticmethod
    def get_used_params():
        return ['subsets_size']


class PoincareSD(CacheOnlyFeature):
    @classmethod
    def _compute(cls, data, params):
        """
        Calculates Poincare SD 1 and 2
        @param data: DataSeries object
        @type data: DataSeries
        @param params: Unused
        @return: Data to cache: (SD1, SD2)
        @rtype: (array, array)
        """
        xd, yd = np.array(list(data[:-1])), np.array(list(data[1:]))
        sd1 = np.std((xd - yd) / np.sqrt(2.0))
        sd2 = np.std((xd + yd) / np.sqrt(2.0))
        return sd1, sd2

    @staticmethod
    def get_used_params():
        return []
