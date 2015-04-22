# coding=utf-8
import spectrum

__author__ = 'AleB'

import numpy as np
from scipy import signal

from pyHRV.Utility import ordered_subsets, interpolate_ibi
from pyHRV.DataSeries import DataSeries
from pyHRV.indexes.BaseFeatures import CacheOnlyFeature
from pyHRV.PyHRVSettings import MainSettings as Sett


class FFTCalc(CacheOnlyFeature):
    @classmethod
    def _calculate_data(cls, data, **kwargs):
        """
        Calculates the FFT data to cache
        @param data: DataSeries object
        @type data: DataSeries
        @param interp_freq: Frequency for the interpolation before the pow. spec. estimation.
        @return: Data to cache: (bands, powers)
        @rtype: (array, array)
        """
        if 'interp_freq' not in kwargs or kwargs['interp_freq'] is None:
            kwargs['interp_freq'] = Sett.default_interpolation_freq
        rr_interp, ignored = interpolate_ibi(data.series, kwargs['interp_freq'])  # TODO 2 Andrea: change interp. type
        interp_freq = kwargs['interp_freq']
        hw = np.hamming(len(rr_interp))

        frame = rr_interp * hw
        frame = frame - np.mean(frame)

        spec_tmp = np.absolute(np.fft.fft(frame)) ** 2  # FFT
        powers = spec_tmp[0:(np.ceil(len(spec_tmp) / 2))]  # Only positive half of spectrum
        bands = np.linspace(start=0, stop=interp_freq / 2, num=len(powers), endpoint=True)  # frequencies vector
        return bands, powers

    def _get_used_params(self, **kwargs):
        return [kwargs['interp_freq']]


class PSDLombscargleCalc(CacheOnlyFeature):
    @classmethod
    def _calculate_data(cls, data, **kwargs):
        """
        Calculates the PSD data to cache using the Lombscargle algorithm
        @param data: DataSeries object
        @type data: DataSeries
        @param interp_freq: Frequency for the interpolation before the pow. spec. estimation.
        @return: Data to cache: (bands, powers, total_power)
        @rtype: (array, array, float)
        """
        if 'interp_freq' not in kwargs or kwargs['interp_freq'] is None:
            kwargs['interp_freq'] = Sett.default_interpolation_freq
        if Sett.remove_mean:
            data = data - np.mean(data)
        t = np.cumsum(data)
        # TODO 2 Andrea: is it an interpolation frequency?
        bands = np.linspace(start=0, stop=kwargs['interp_freq'] / 2, num=max(128, len(data)), endpoint=True)
        bands = bands[1:]
        powers = np.sqrt(4 * (signal.lombscargle(t, data, bands) / len(data)))

        return bands, powers / np.max(powers), sum(powers) / len(powers)

    def _get_used_params(self, **kwargs):
        return [kwargs['interp_freq']]


class PSDFFTCalc(CacheOnlyFeature):
    @classmethod
    def _calculate_data(cls, data, **kwargs):
        """
        Calculates the PSD data to cache using the fft algorithm
        @param data: DataSeries object
        @type data: DataSeries
        @param interp_freq: Frequency for the interpolation before the pow. spec. estimation.
        @return: Data to cache: (bands, powers, total_power)
        @rtype: (array, array, float)
        """
        if 'interp_freq' not in kwargs or kwargs['interp_freq'] is None:
            kwargs['interp_freq'] = Sett.default_interpolation_freq
        data_interp, t_interp = interpolate_ibi(data, kwargs['interp_freq'])  # TODO 2 Andrea: change interp. type
        if Sett.remove_mean:
            data_interp = data_interp - np.mean(data_interp)

        hw = np.hamming(len(data_interp))
        frame = data_interp * hw
        spec_tmp = np.absolute(np.fft.fft(frame)) ** 2  # FFT
        powers = spec_tmp[0:(np.ceil(len(spec_tmp) / 2))]

        bands = np.linspace(start=0, stop=kwargs['interp_freq'] / 2, num=len(powers), endpoint=True)

        return bands, powers / np.max(powers), sum(powers) / len(powers)

    def _get_used_params(self, **kwargs):
        return [kwargs['interp_freq']]


class PSDWelchLinspaceCalc(CacheOnlyFeature):
    @classmethod
    def _calculate_data(cls, data, **kwargs):
        """
        Calculates the PSD data to cache using the welch algorithm, uses linspace bands distribution
        @param data: DataSeries object
        @type data: DataSeries
        @param interp_freq: Frequency for the interpolation before the pow. spec. estimation.
        @return: Data to cache: (bands, powers, total_power)
        @rtype: (array, array, float)
        """
        if 'interp_freq' not in kwargs or kwargs['interp_freq'] is None:
            kwargs['interp_freq'] = Sett.default_interpolation_freq
        data_interp, t_interp = interpolate_ibi(data, kwargs['interp_freq'])  # TODO 2 Andrea: change interp. type
        if Sett.remove_mean:
            data_interp = data_interp - np.mean(data_interp)
        bands_w, powers = signal.welch(data_interp, kwargs['interp_freq'], nfft=max(128, len(data_interp)))
        bands = np.linspace(start=0, stop=kwargs['interp_freq'] / 2, num=len(powers), endpoint=True)
        return bands, powers / np.max(powers), sum(powers) / len(powers)

    def _get_used_params(self, **kwargs):
        return [kwargs['interp_freq']]


class PSDWelchLibCalc(CacheOnlyFeature):
    @classmethod
    def _calculate_data(cls, data, **kwargs):
        """
        Calculates the PSDWelch data to cache, uses algorithms bands distribution
        @param data: DataSeries object
        @type data: DataSeries
        @param interp_freq: Frequency for the interpolation before the pow. spec. estimation.
        @return: Data to cache: (bands, powers, total_power)
        @rtype: (array, array, float)
        """
        if 'interp_freq' not in kwargs or kwargs['interp_freq'] is None:
            kwargs['interp_freq'] = Sett.default_interpolation_freq
        rr_interp, bt_interp = interpolate_ibi(data, kwargs['interp_freq'])  # TODO 2 Andrea: change interp. type
        bands, powers = signal.welch(rr_interp, kwargs['interp_freq'], nfft=max(128, len(rr_interp)))
        powers = np.sqrt(powers)
        return bands, powers / np.max(powers), sum(powers) / len(powers)

    def _get_used_params(self, **kwargs):
        return [kwargs['interp_freq']]


class PSDAr1Calc(CacheOnlyFeature):
    @classmethod
    def _calculate_data(cls, data, **kwargs):
        """
        Calculates the PSD data to cache using the ar_1 algorithm
        @param data: DataSeries object
        @type data: DataSeries
        @param interp_freq: Frequency for the interpolation before the pow. spec. estimation.
        @return: Data to cache: (bands, powers, total_power)
        @rtype: (array, array, float)
        """
        if 'interp_freq' not in kwargs or kwargs['interp_freq'] is None:
            kwargs['interp_freq'] = Sett.default_interpolation_freq
        data_interp, t_interp = interpolate_ibi(data, kwargs['interp_freq'])  # TODO 2 Andrea: change interp. type
        if Sett.remove_mean:
            data_interp = data_interp - np.mean(data_interp)

        p = spectrum.Periodogram(data_interp, sampling=kwargs['interp_freq'], NFFT=max(128, len(data_interp)))
        p()
        powers = p.get_converted_psd('onesided')
        bands = np.linspace(start=0, stop=kwargs['interp_freq'] / 2, num=len(powers), endpoint=True)

        return bands, powers / np.max(powers), sum(powers) / len(powers)

    def _get_used_params(self, **kwargs):
        return [kwargs['interp_freq']]


class PSDAr2Calc(CacheOnlyFeature):
    @classmethod
    def _calculate_data(cls, data, **kwargs):
        """
        Calculates the PSD data to cache using the ar_2 algorithm
        @param data: DataSeries object
        @type data: DataSeries
        @param interp_freq: Frequency for the interpolation before the pow. spec. estimation.
        @return: Data to cache: (bands, powers, total_power)
        @rtype: (array, array, float)
        """
        if 'interp_freq' not in kwargs or kwargs['interp_freq'] is None:
            kwargs['interp_freq'] = Sett.default_interpolation_freq
        powers = []

        data_interp, t_interp = interpolate_ibi(data, kwargs['interp_freq'])  # TODO 2 Andrea: change interp. type
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

        bands = np.linspace(start=0, stop=kwargs['interp_freq'] / 2, num=len(powers), endpoint=True)

        return bands, powers / np.max(powers), sum(powers) / len(powers)

    def _get_used_params(self, **kwargs):
        return [kwargs['interp_freq']]


class Histogram(CacheOnlyFeature):
    @classmethod
    def _calculate_data(cls, data, **kwargs):
        """
        Calculates the Histogram data to cache
        @param data: DataSeries object
        @type data: DataSeries
        @param histogram_bins: Histogram bins
        @return: Data to cache: (hist, bin_edges)
        @rtype: (array, array)
        """
        if 'histogram_bins' not in kwargs or kwargs['histogram_bins'] is None:
            kwargs['histogram_bins'] = Sett.cache_histogram_bins
        return np.histogram(data, kwargs['histogram_bins'])

    def _get_used_params(self, **kwargs):
        return [kwargs['histogram_bins']]


class HistogramMax(CacheOnlyFeature):
    @classmethod
    def _calculate_data(cls, data, **kwargs):
        """
        Calculates the Histogram's max value
        @param data: DataSeries object
        @type data: DataSeries
        @param histogram_bins: Histogram bins
        @return: Data to cache: (hist, bin_edges)
        @rtype: (array, array)
        """
        h, b = Histogram.get(data, kwargs['histogram_bins'])
        return np.max(h)  # TODO 2 Andrea: max h or b(max h)??

    def _get_used_params(self, **kwargs):
        return [kwargs['histogram_bins']]


class Diff(CacheOnlyFeature):
    @classmethod
    def _calculate_data(cls, data, **kwargs):
        """
        Calculates the differences between consecutive values
        @param data: DataSeries object
        @type data: DataSeries
        @param params: Unused
        @return: Data to cache: diff
        @rtype: array
        """
        return np.diff(np.array(data))

    def _get_used_params(self, **kwargs):
        return []


class StandardDeviation(CacheOnlyFeature):
    @classmethod
    def _calculate_data(cls, data, **kwargs):
        """
        Calculates the standard deviation data
        @param data: DataSeries object
        @type data: DataSeries
        @param params: Unused
        @return: Data to cache: st. dev.
        @rtype: array
        """
        return np.std(np.array(data))

    def _get_used_params(self, **kwargs):
        return []


class OrderedSubsets2(CacheOnlyFeature):
    @classmethod
    def _calculate_data(cls, data, **kwargs):
        """
        Calculates the the vector of the sequences of length 2 of the data
        @param data: DataSeries object
        @type data: DataSeries
        @param params: Unused
        @return: Data to cache: Tokens vector (2)
        @rtype: array
        """
        return ordered_subsets(data, 2)

    def _get_used_params(self, **kwargs):
        return []


class OrderedSubsets3(CacheOnlyFeature):
    @classmethod
    def _calculate_data(cls, data, **kwargs):
        """
        Calculates the the vector of the sequences of length 2 of the data
        @param data: DataSeries object
        @type data: DataSeries
        @param params: Unused
        @return: Data to cache: Tokens vector (3)
        @rtype: array
        """
        return ordered_subsets(data, 3)

    def _get_used_params(self, **kwargs):
        return []


class PoincareSD(CacheOnlyFeature):
    @classmethod
    def _calculate_data(cls, data, **kwargs):
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

    def _get_used_params(self, **kwargs):
        return []
