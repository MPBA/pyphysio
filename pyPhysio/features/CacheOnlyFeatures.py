# coding=utf-8
from __future__ import division

# coding=utf-8
import spectrum

__author__ = 'AleB'

import numpy as np
from scipy import signal

from ..Utility import interpolate_ibi
from ..BaseFeature import Feature


class CacheOnlyFeature(Feature):
    """
    This is the base class for the generic features.
    """

    def __init__(self, params=None, _kwargs=None):
        super(CacheOnlyFeature, self).__init__(params, _kwargs)

    @classmethod
    def algorithm(cls, data, params):
        """
        Placeholder for the subclasses
        @raise NotImplementedError: Ever
        """
        raise NotImplementedError(cls.__name__ + " is a CacheOnlyFeature but it is not implemented.")


class FFTCalc(CacheOnlyFeature):
    @classmethod
    def algorithm(cls, data, params):
        assert 'interp_freq' in params, "This feature needs the parameter 'interp_freq' [1/time_unit]."
        rr_interp, ignored = interpolate_ibi(data.series, params['interp_freq'])  # TODO 2 Andrea: change interp. type
        interp_freq = params['interp_freq']
        hw = np.hamming(len(rr_interp))

        frame = rr_interp * hw
        frame = frame - np.mean(frame)

        spec_tmp = np.absolute(np.fft.fft(frame)) ** 2  # FFT
        powers = spec_tmp[0:(np.ceil(len(spec_tmp) / 2))]  # Only positive half of spectrum
        bands = np.linspace(start=0, stop=interp_freq / 2, num=len(powers))  # frequencies vector
        return bands, powers

    @classmethod
    def get_used_params(cls):
        return ['interp_freq']


class PSDLombscargleCalc(CacheOnlyFeature):
    @classmethod
    def algorithm(cls, data, params):
        """
        Calculates the PSD data to cache using the Lombscargle algorithm
        @return: (bands, powers, total_power)
        @rtype: (array, array, float)
        """
        # TODO 5 Andrea: is it an interpolation frequency?
        assert 'lombscargle_stop' in params, "This feature needs the parameter 'lombscargle_stop'."
        if 'remove_mean' not in params:
            params['remove_mean'] = False
        if params['remove_mean']:
            data = data - np.mean(data)
        t = np.cumsum(data)

        # stop : scalar
        #     The end value of the sequence, unless endpoint is set to False. In that case, the sequence consists of
        #     all but the last of num + 1 evenly spaced samples, so that stop is excluded. Note that the step size
        #     changes when endpoint is False.

        bands = np.linspace(start=0, stop=params['lombscargle_stop'] / 2, num=max(128, len(data)))
        bands = bands[1:]
        powers = np.sqrt(4 * (signal.lombscargle(t, data, bands) / len(data)))

        return bands, powers / np.max(powers), sum(powers) / len(powers)

    @classmethod
    def get_used_params(cls):
        return ['lombscargle_stop', 'remove_mean']


class PSDFFTCalc(CacheOnlyFeature):
    @classmethod
    def algorithm(cls, data, params):
        """
        Calculates the PSD data to cache using the fft algorithm
        @return: (bands, powers, total_power)
        @rtype: (array, array, float)
        """
        assert 'interp_freq' in params, "This feature needs the parameter 'interp_freq' [1/time_unit]."
        if 'remove_mean' not in params:
            params['remove_mean'] = False
        data_interp, t_interp = interpolate_ibi(data, params['interp_freq'])  # TODO 6: change interp. type
        if params['remove_mean']:
            data_interp = data_interp - np.mean(data_interp)

        hw = np.hamming(len(data_interp))
        frame = data_interp * hw
        spec_tmp = np.absolute(np.fft.fft(frame)) ** 2  # FFT
        powers = spec_tmp[0:(np.ceil(len(spec_tmp) / 2))]

        bands = np.linspace(start=0, stop=params['interp_freq'] / 2, num=len(powers))

        return bands, powers / np.max(powers), sum(powers) / len(powers)

    @classmethod
    def get_used_params(cls):
        return ['interp_freq', 'remove_mean']


class PSDWelchLinspaceCalc(CacheOnlyFeature):
    @classmethod
    def algorithm(cls, data, params):
        """
        Calculates the PSD data to cache using the welch algorithm, uses 'linspace' bands distribution
        @return: (bands, powers, total_power)
        @rtype: (array, array, float)
        """
        assert 'interp_freq' in params, "This feature needs the parameter 'interp_freq' [1/time_unit]."
        if 'remove_mean' not in params:
            params['remove_mean'] = False
        data_interp, t_interp = interpolate_ibi(data, params['interp_freq'])  # TODO 6: change interp. type
        if params['remove_mean']:
            data_interp = data_interp - np.mean(data_interp)
        bands_w, powers = signal.welch(data_interp, params['interp_freq'], nfft=max(128, len(data_interp)))
        bands = np.linspace(start=0, stop=params['interp_freq'] / 2, num=len(powers))
        return bands, powers / np.max(powers), sum(powers) / len(powers)

    @classmethod
    def get_used_params(cls):
        return ['interp_freq', 'remove_mean']


class PSDWelchLibCalc(CacheOnlyFeature):
    @classmethod
    def algorithm(cls, data, params):
        """
        Calculates the PSDWelch data to cache, uses algorithms bands distribution
        @return: (bands, powers, total_power)
        @rtype: (array, array, float)
        """
        assert 'interp_freq' in params, "This feature needs the parameter 'interp_freq' [1/time_unit]."
        rr_interp, bt_interp = interpolate_ibi(data, params['interp_freq'])  # TODO 6: change interp. type
        bands, powers = signal.welch(rr_interp, params['interp_freq'], nfft=max(128, len(rr_interp)))
        powers = np.sqrt(powers)
        return bands, powers / np.max(powers), sum(powers) / len(powers)

    @classmethod
    def get_used_params(cls):
        return ['interp_freq']


class PSDAr1Calc(CacheOnlyFeature):
    @classmethod
    def algorithm(cls, data, params):
        """
        Calculates the PSD data to cache using the ar_1 algorithm
        @return: Data to cache: (bands, powers, total_power)
        @rtype: (array, array, float)
        """
        assert 'interp_freq' in params, "This feature needs the parameter 'interp_freq' [1/time_unit]."
        if 'remove_mean' not in params:
            params['remove_mean'] = False
        data_interp, t_interp = interpolate_ibi(data, params['interp_freq'])  # TODO 6: change interp. type
        # TODO Andrea: remove_mean WAS after interp, is it ok?
        if params['remove_mean']:
            data_interp = data_interp - np.mean(data_interp)

        p = spectrum.Periodogram(data_interp, sampling=params['interp_freq'], NFFT=max(128, len(data_interp)))
        p()
        powers = p.get_converted_psd('onesided')
        bands = np.linspace(start=0, stop=params['interp_freq'] / 2, num=len(powers))

        return bands, powers / np.max(powers), sum(powers) / len(powers)

    @classmethod
    def get_used_params(cls):
        return ['interp_freq', 'remove_mean']


class PSDAr2Calc(CacheOnlyFeature):
    @classmethod
    def algorithm(cls, data, params):
        """
        Calculates the PSD data to cache using the ar_2 algorithm
        @return: (bands, powers, total_power)
        @rtype: (array, array, float)
        """
        assert 'interp_freq' in params, "This feature needs the parameter 'interp_freq' [1/time_unit]."
        assert 'ar_2_max_order' in params, "This feature needs the parameter 'ar_2_max_order'."
        if 'remove_mean' not in params:
            params['remove_mean'] = False
        data_interp, t_interp = interpolate_ibi(data, params['interp_freq'])  # TODO 6: change interp. type
        if params['remove_mean']:
            data_interp = data_interp - np.mean(data_interp)
        powers = []

        orders = range(1, params['ar_2_max_order'] + 1)
        for order in orders:
            try:
                ar, p, k = spectrum.aryule(data_interp, order=order)
            except AssertionError:
                ar = 1
                print("Error in ar_2 psd ayrule, assumed ar=1")
            powers = spectrum.arma2psd(ar, NFFT=max(128, len(data_interp)))
            powers = powers[0: np.ceil(len(powers) / 2)]
        else:
            print("Error in ar_2 psd, orders=0, empty powers")

        bands = np.linspace(start=0, stop=params['interp_freq'] / 2, num=len(powers))

        return bands, powers / np.max(powers), sum(powers) / len(powers)

    @classmethod
    def get_used_params(cls):
        return ['interp_freq', 'ar_2_max_order', 'remove_mean']


class Histogram(CacheOnlyFeature):
    @classmethod
    def algorithm(cls, data, params):
        """
        Calculates the Histogram data to cache
        @return: (values, bins)
        @rtype: (array, array)
        """
        if 'histogram_bins' not in params or params['histogram_bins'] is None:
            params['histogram_bins'] = 100
        return np.histogram(data, params['histogram_bins'])

    @classmethod
    def get_used_params(cls):
        return ['histogram_bins']


class HistogramMax(CacheOnlyFeature):
    @classmethod
    def algorithm(cls, data, params):
        """
        Calculates the Histogram's max value
        @return: (values, bins)
        @rtype: (array, array)
        """
        h, b = Histogram.get(data, params)
        return np.max(h)  # TODO 2 Andrea: max h or b(max h)??

    @classmethod
    def get_used_params(cls):
        return Histogram.get_used_params()


class OrderedSubsets(CacheOnlyFeature):
    @classmethod
    def algorithm(cls, data, params):
        """
        Calculates the the vector of the sequences of length 'subset_size' of the data
        @return: Data array with shape (l - n + 1, n) having l=len(data) and n=subset_size
        @rtype: array
        """
        assert 'subset_size' in params, "This feature needs the parameter 'subset_size'."
        n = params['subset_size']
        num = len(data) - n + 1
        if num > 0:
            emb = np.zeros([num, n])
            for i in xrange(num):
                emb[i, :] = data[i:i + n]
            return emb
        else:
            return []

    @classmethod
    def get_used_params(cls):
        return ['subsets_size']


class PoincareSD(CacheOnlyFeature):
    @classmethod
    def algorithm(cls, data, params):
        """
        Calculates Poincare SD 1 and 2
        @return: (SD1, SD2)
        @rtype: (array, array)
        """
        xd, yd = np.array(list(data[:-1])), np.array(list(data[1:]))
        sd1 = np.std((xd - yd) / np.sqrt(2.0))
        sd2 = np.std((xd + yd) / np.sqrt(2.0))
        return sd1, sd2

