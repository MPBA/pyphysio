# coding=utf-8
from __future__ import division

import spectrum
from scipy import signal
from ..Utility import interpolate_ibi as _interpolate_ibi
from ..BaseIndicator import Indicator as _Indicator
from ..indicators.SupportValues import SumSV as _SumSV, LengthSV as _LengthSV, DiffsSV as _DiffsSV, \
    MedianSV as _MedianSV
import numpy as _np

__author__ = 'AleB'


class FFTCalc(_Indicator):
    @classmethod
    def algorithm(cls, data, params):
        assert 'interp_freq' in params, "This feature needs the parameter 'interp_freq' [1/time_unit]."
        rr_interp, ignored = _interpolate_ibi(data.series, params['interp_freq'])  # TODO 2 Andrea: change interp. type
        interp_freq = params['interp_freq']
        hw = _np.hamming(len(rr_interp))

        frame = rr_interp * hw
        frame = frame - _np.mean(frame)
        f_fft = _np.fft.fft(frame)
        spec_tmp = _np.absolute(f_fft) ** 2  # FFT
        p_half = _np.ceil(len(f_fft) / 2)
        powers = spec_tmp[0:p_half]  # Only positive half of spectrum
        bands = _np.linspace(start=0, stop=interp_freq / 2, num=len(p_half))  # frequencies vector
        return bands, powers

    @classmethod
    def get_used_params(cls):
        return ['interp_freq']


class PSDLombscargleCalc(_Indicator):
    @classmethod
    def algorithm(cls, data, params):
        """
        Calculates the PSD data to cache using the Lombscargle algorithm
        @return: (bands, powers, total_power)
        @rtype: (array, array, float)
        :param params:
        :param data:
        """
        # TODO 5 Andrea: is it an interpolation frequency?
        assert 'lombscargle_stop' in params, "This feature needs the parameter 'lombscargle_stop'."
        if 'remove_mean' not in params:
            params['remove_mean'] = False
        if params['remove_mean']:
            data = data - _np.mean(data)
        t = _np.cumsum(data)

        # stop : scalar
        # The end value of the sequence, unless endpoint is set to False. In that case, the sequence consists of
        #     all but the last of num + 1 evenly spaced samples, so that stop is excluded. Note that the step size
        #     changes when endpoint is False.

        bands = _np.linspace(start=0, stop=params['lombscargle_stop'] / 2, num=max(128, len(data)))
        bands = bands[1:]
        powers = _np.sqrt(4 * (signal.lombscargle(t, data, bands) / len(data)))

        return bands, powers / _np.max(powers), sum(powers) / len(powers)

    @classmethod
    def get_used_params(cls):
        return ['lombscargle_stop', 'remove_mean']


class PSDFFTCalc(_Indicator):
    @classmethod
    def algorithm(cls, data, params):
        """
        Calculates the PSD data to cache using the fft algorithm
        @return: (bands, powers, total_power)
        @rtype: (array, array, float)
        :param params:
        :param data:
        """
        assert 'interp_freq' in params, "This feature needs the parameter 'interp_freq' [1/time_unit]."
        if 'remove_mean' not in params:
            params['remove_mean'] = False
        data_interp, t_interp = _interpolate_ibi(data, params['interp_freq'])  # TODO 6: change interp. type
        if params['remove_mean']:
            data_interp = data_interp - _np.mean(data_interp)

        hw = _np.hamming(len(data_interp))
        frame = data_interp * hw
        spec_tmp = _np.absolute(_np.fft.fft(frame)) ** 2  # FFT
        powers = spec_tmp[0:(_np.ceil(len(spec_tmp) / 2))]

        bands = _np.linspace(start=0, stop=params['interp_freq'] / 2, num=len(powers))

        return bands, powers / _np.max(powers), sum(powers) / len(powers)

    @classmethod
    def get_used_params(cls):
        return ['interp_freq', 'remove_mean']


class PSDWelchLinspaceCalc(_Indicator):
    @classmethod
    def algorithm(cls, data, params):
        """
        Calculates the PSD data to cache using the welch algorithm, uses 'linspace' bands distribution
        @return: (bands, powers, total_power)
        @rtype: (array, array, float)
        :param params:
        :param data:
        """
        assert 'interp_freq' in params, "This feature needs the parameter 'interp_freq' [1/time_unit]."
        if 'remove_mean' not in params:
            params['remove_mean'] = False
        data_interp, t_interp = _interpolate_ibi(data, params['interp_freq'])  # TODO 6: change interp. type
        if params['remove_mean']:
            data_interp = data_interp - _np.mean(data_interp)
        bands_w, powers = signal.welch(data_interp, params['interp_freq'], nfft=max(128, len(data_interp)))
        bands = _np.linspace(start=0, stop=params['interp_freq'] / 2, num=len(powers))
        return bands, powers / _np.max(powers), sum(powers) / len(powers)

    @classmethod
    def get_used_params(cls):
        return ['interp_freq', 'remove_mean']


class PSDWelchLibCalc(_Indicator):
    @classmethod
    def algorithm(cls, data, params):
        """
        Calculates the PSDWelch data to cache, uses algorithms bands distribution
        @return: (bands, powers, total_power)
        @rtype: (array, array, float)
        """
        assert 'interp_freq' in params, "This feature needs the parameter 'interp_freq' [1/time_unit]."
        rr_interp, bt_interp = _interpolate_ibi(data, params['interp_freq'])  # TODO 6: change interp. type
        bands, powers = signal.welch(rr_interp, params['interp_freq'], nfft=max(128, len(rr_interp)))
        powers = _np.sqrt(powers)
        return bands, powers / _np.max(powers), sum(powers) / len(powers)

    @classmethod
    def get_used_params(cls):
        return ['interp_freq']


class PSDAr1Calc(_Indicator):
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
        data_interp, t_interp = _interpolate_ibi(data, params['interp_freq'])  # TODO 6: change interp. type
        # TODO Andrea: remove_mean WAS after interp, is it ok?
        if params['remove_mean']:
            data_interp = data_interp - _np.mean(data_interp)

        p = spectrum.Periodogram(data_interp, sampling=params['interp_freq'], NFFT=max(128, len(data_interp)))
        p()
        powers = p.get_converted_psd('onesided')
        bands = _np.linspace(start=0, stop=params['interp_freq'] / 2, num=len(powers))

        return bands, powers / _np.max(powers), sum(powers) / len(powers)

    @classmethod
    def get_used_params(cls):
        return ['interp_freq', 'remove_mean']


class PSDAr2Calc(_Indicator):
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
        data_interp, t_interp = _interpolate_ibi(data, params['interp_freq'])  # TODO 6: change interp. type
        if params['remove_mean']:
            data_interp = data_interp - _np.mean(data_interp)
        powers = []

        orders = range(1, params['ar_2_max_order'] + 1)
        for order in orders:
            try:
                ar, p, k = spectrum.aryule(data_interp, order=order)
            except AssertionError:
                ar = 1
                print("Error in ar_2 psd ayrule, assumed ar=1")
            powers = spectrum.arma2psd(ar, NFFT=max(128, len(data_interp)))
            powers = powers[0: _np.ceil(len(powers) / 2)]
        else:
            print("Error in ar_2 psd, orders=0, empty powers")

        bands = _np.linspace(start=0, stop=params['interp_freq'] / 2, num=len(powers))

        return bands, powers / _np.max(powers), sum(powers) / len(powers)

    @classmethod
    def get_used_params(cls):
        return ['interp_freq', 'ar_2_max_order', 'remove_mean']


class Histogram(_Indicator):
    @classmethod
    def algorithm(cls, data, params):
        """
        Calculates the Histogram data to cache
        @return: (values, bins)
        @rtype: (array, array)
        """
        if 'histogram_bins' not in params or params['histogram_bins'] is None:
            params['histogram_bins'] = 100
        return _np.histogram(data, params['histogram_bins'])

    @classmethod
    def get_used_params(cls):
        return ['histogram_bins']


class HistogramMax(_Indicator):
    @classmethod
    def algorithm(cls, data, params):
        """
        Calculates the Histogram's max value
        @return: (values, bins)
        @rtype: (array, array)
        """
        h, b = Histogram.get(data, params)
        return _np.max(h)  # TODO 2 Andrea: max h or b(max h)??

    @classmethod
    def get_used_params(cls):
        return Histogram.get_used_params()


class OrderedSubsets(_Indicator):
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
            emb = _np.zeros([num, n])
            for i in xrange(num):
                emb[i, :] = data[i:i + n]
            return emb
        else:
            return []

    @classmethod
    def get_used_params(cls):
        return ['subsets_size']


class PoincareSD(_Indicator):
    @classmethod
    def algorithm(cls, data, params):
        """
        Calculates Poincare SD 1 and 2
        @return: (SD1, SD2)
        @rtype: (array, array)
        """
        xd, yd = _np.array(list(data[:-1])), _np.array(list(data[1:]))
        sd1 = _np.std((xd - yd) / _np.sqrt(2.0))
        sd2 = _np.std((xd + yd) / _np.sqrt(2.0))
        return sd1, sd2


class Mean(_Indicator):
    """
    Calculates the average value of the data.
    """

    def __init__(self, params=None, **kwargs):
        super(Mean, self).__init__(params, **kwargs)

    @classmethod
    def algorithm(cls, data, params):
        return _np.mean(data)

    @classmethod
    def compute_on(cls, state):
        return state[_SumSV].value / float(state[_LengthSV].value)

    @classmethod
    def required_sv(cls):
        return [_SumSV, _LengthSV]


class Median(_Indicator):
    """
    Calculates the median of the data series.
    """

    def __init__(self, params=None, **kwargs):
        super(Median, self).__init__(params, **kwargs)

    @classmethod
    def algorithm(cls, data, params):
        return _np.median(data)

    @classmethod
    def required_sv(cls):
        return [_MedianSV]

    @classmethod
    def compute_on(cls, state):
        return state[_MedianSV].value


class SD(_Indicator):
    """
    Calculates the standard deviation of the data series.
    """

    def __init__(self, params=None, **kwargs):
        super(SD, self).__init__(params, **kwargs)

    @classmethod
    def algorithm(cls, data, params):
        return _np.std(data)


class PNNx(_Indicator):
    """
    Calculates the relative frequency (0.0-1.0) of pairs of consecutive IBIs in the data series
    where the difference between the two values is greater than the parameter (threshold).
    """

    def __init__(self, params=None, **kwargs):
        super(PNNx, self).__init__(params, **kwargs)

    @classmethod
    def algorithm(cls, data, params):
        if cls == PNNx:
            assert 'threshold' in params, "Need the parameter 'threshold'."
            px = params
        else:
            px = params.copy()
            px.update({'threshold': cls.threshold()})
        return NNx.algorithm(data, px) / float(len(data))

    @staticmethod
    def threshold():
        raise NotImplementedError()

    @classmethod
    def required_sv(cls):
        return NNx.required_sv()

    @classmethod
    def compute_on(cls, state):
        return NNx.compute_on(state, cls.threshold()) / state[_LengthSV].value


from ..filters.Filters import Diff as _Diff


class NNx(_Indicator):
    """
    Calculates number of pairs of consecutive values in the data where the difference between is greater than the given
    parameter (threshold).
    """

    def __init__(self, params=None, **kwargs):
        super(NNx, self).__init__(params, **kwargs)

    @classmethod
    def algorithm(cls, data, params):
        if cls == NNx:
            assert 'threshold' in params, "Need the parameter 'threshold'."
            th = params['threshold']
        else:
            th = cls.threshold()
        diff = _Diff.get(data)
        return sum(1.0 for x in diff if x > th)

    @staticmethod
    def get_used_params(**kwargs):
        return ['threshold']

    @staticmethod
    def threshold():
        raise NotImplementedError()

    @classmethod
    def required_sv(cls):
        return [_DiffsSV]

    @classmethod
    def compute_on(cls, state, threshold=None):
        if threshold is None:
            threshold = cls.threshold()
        return sum(1 for x in state[_DiffsSV].value if x > threshold)


class PNN10(PNNx):
    """
    Calculates the relative frequency (0.0-1.0) of the pairs of consecutive values in the data where the difference is
    greater than 10.
    """

    def __init__(self, params=None, **kwargs):
        super(PNN10, self).__init__(params, **kwargs)

    @staticmethod
    def threshold():
        return 10


class PNN25(PNNx):
    """
    Calculates the relative frequency (0.0-1.0) of the pairs of consecutive values in the data where the difference is
    greater than 25.
    """

    def __init__(self, params=None, **kwargs):
        super(PNN25, self).__init__(params, **kwargs)

    @staticmethod
    def threshold():
        return 25


class PNN50(PNNx):
    """
    Calculates the relative frequency (0.0-1.0) of the pairs of consecutive values in the data where the difference is
    greater than 50.
    """

    def __init__(self, params=None, **kwargs):
        super(PNN50, self).__init__(params, **kwargs)

    @staticmethod
    def threshold():
        return 50


class NN10(NNx):
    """
    Calculates number of pairs of consecutive values in the data where the difference is greater than 10.
    """

    def __init__(self, params=None, **kwargs):
        super(NN10, self).__init__(params, **kwargs)

    @staticmethod
    def threshold():
        return 10


class NN25(NNx):
    """
    Calculates number of pairs of consecutive values in the data where the difference is greater than 25.
    """

    def __init__(self, params=None, **kwargs):
        super(NN25, self).__init__(params, **kwargs)

    @staticmethod
    def threshold():
        return 25


class NN50(NNx):
    """
    Calculates number of pairs of consecutive values in the data where the difference is greater than 50.
    """

    def __init__(self, params=None, **kwargs):
        super(NN50, self).__init__(params, **kwargs)

    @staticmethod
    def threshold():
        return 50


class RMSSD(_Indicator):
    """
    Calculates the square root of the mean of the squared differences.
    """

    def __init__(self, params=None, **kwargs):
        super(RMSSD, self).__init__(params, **kwargs)

    @classmethod
    def algorithm(cls, data, params):
        diff = _Diff.get(data)
        return _np.sqrt(sum(diff ** 2) / len(diff))


class DiffSD(_Indicator):
    """Calculates the standard deviation of the differences between each value and its next."""

    def __init__(self, params=None, **kwargs):
        super(DiffSD, self).__init__(params, **kwargs)

    @classmethod
    def algorithm(cls, data, params):
        diff = _Diff.get(data)
        return _np.std(diff)


# TODO: fix documentation
class Triang(_Indicator):
    """Calculates the Triangular index that is the ratio between the number of samples and the number of samples in the
    highest bin of the data's 100 bin histogram."""

    def __init__(self, params=None, **kwargs):
        super(Triang, self).__init__(params, **kwargs)

    @classmethod
    def algorithm(cls, data, params):
        h, b = HistogramMax.get(data, histogram_bins=100)
        return len(data) / _np.max(h)  # TODO: check if the formula is the right one or use the HistogramMax


# TODO: fix documentation
class TINN(_Indicator):
    """Calculates the difference between two histogram-related indicators."""

    def __init__(self, params=None, **kwargs):
        super(TINN, self).__init__(params, **kwargs)

    @classmethod
    def algorithm(cls, data, params):
        hist, bins = Histogram.get(data, histogram_bins=100)
        max_x = HistogramMax.get(data)
        hist_left = _np.array(hist[0:_np.argmax(hist)])
        ll = len(hist_left)
        hist_right = _np.array(hist[_np.argmax(hist):-1])
        rl = len(hist_right)

        y_left = _np.array(_np.linspace(0, max_x, ll))

        minx = _np.Inf
        pos = 0
        for i in range(len(hist_left) - 1):
            curr_min = _np.sum((hist_left - y_left) ** 2)
            if curr_min < minx:
                minx = curr_min
                pos = i
            y_left[i] = 0
            y_left[i + 1:] = _np.linspace(0, max_x, ll - i - 1)

        n = bins[pos - 1]

        y_right = _np.array(_np.linspace(max_x, 0, rl))
        minx = _np.Inf
        pos = 0
        for i in range(rl, 1, -1):
            curr_min = _np.sum((hist_right - y_right) ** 2)
            if curr_min < minx:
                minx = curr_min
                pos = i
            y_right[i - 1] = 0
            y_right[0:i - 2] = _np.linspace(max_x, 0, i - 2)

        m = bins[_np.argmax(hist) + pos + 1]

        return m - n


class InBand(_Indicator):
    def __init__(self, params=None, **kwargs):
        super(InBand, self).__init__(params, **kwargs)

    @classmethod
    def algorithm(cls, data, params):
        assert 'freq_min' in params, "Need the parameter 'freq_min' as the lower bound of the band."
        assert 'freq_max' in params, "Need the parameter 'freq_max' as the higher bound of the band."

        if 'psd_method' not in params:
            params.update({'psd_method': PSDWelchLibCalc})

        freq, spec, total = params['psd_method'].get(data, params)

        return ([freq[i] for i in xrange(len(freq)) if params['freq_min'] <= freq[i] < params['freq_max']],
                [spec[i] for i in xrange(len(spec)) if params['freq_min'] <= freq[i] < params['freq_max']],
                total)

    @classmethod
    def get_used_params(cls):
        return ['freq_max', 'freq_min'] + PSDWelchLibCalc.get_used_params()


class PowerInBand(_Indicator):
    def __init__(self, params=None, **kwargs):
        super(PowerInBand, self).__init__(params, **kwargs)

    @classmethod
    def algorithm(cls, data, params):
        ignore, _pow_band, ignored = InBand.get(data, **params)
        return sum(_pow_band) / len(_pow_band)

    @classmethod
    def get_used_params(cls):
        return InBand.get_used_params()


class PowerInBandNormal(_Indicator):
    def __init__(self, params=None, **kwargs):
        super(PowerInBandNormal, self).__init__(params, **kwargs)

    @classmethod
    def algorithm(cls, data, params):
        ignored, _pow_band, _pow_total = InBand.get(data, params)
        return sum(_pow_band) / len(_pow_band) / _pow_total

    @classmethod
    def get_used_params(cls):
        return InBand.get_used_params()


class PeakInBand(_Indicator):
    def __init__(self, params=None, **kwargs):
        super(PeakInBand, self).__init__(params, **kwargs)

    @classmethod
    def algorithm(cls, data, params):
        _freq_band, _pow_band, ignored = InBand.get(data, params)
        return _freq_band[_np.argmax(_pow_band)]

    @classmethod
    def get_used_params(cls):
        return InBand.get_used_params()


class LFHF(_Indicator):
    def __init__(self, params=None, **kwargs):
        super(LFHF, self).__init__(params, **kwargs)

    @classmethod
    def algorithm(cls, data, params):
        assert 'freq_mid' in params, "Need the parameter 'freq_mid' as the separator between LF and HF."
        par_lf = params.copy()
        par_hf = params.copy()
        par_lf.update({'freq_max': params['freq_mid']})
        par_hf.update({'freq_min': params['freq_mid']})
        return PowerInBand.get(data, par_lf) / PowerInBand.get(data, par_hf)

    @classmethod
    def get_used_params(cls):
        return PowerInBand.get_used_params()


class NormalizedLF(_Indicator):
    """
    Calculates the normalized power value of the LF band (parametrized in the settings) over the LF and HF bands.
    """

    def __init__(self, params=None, **kwargs):
        super(NormalizedLF, self).__init__(params, **kwargs)

    @classmethod
    def algorithm(cls, data, params):
        assert 'freq_mid' in params, "Need the parameter 'freq_mid' as the separator between LF and HF."
        par_lf = params.copy().update({'freq_max': params['freq_mid']})
        par_hf = params.copy().update({'freq_min': params['freq_mid']})
        return PowerInBand.get(data, par_lf) / (PowerInBand.get(data, par_hf) + PowerInBand.get(data, par_lf))

    @classmethod
    def get_used_params(cls):
        return PowerInBand.get_used_params() + ['freq_mid']


class NormalizedHF(_Indicator):
    """
    Calculates the normalized power value of the HF band (parametrized in the settings) over the LF and HF bands.
    """

    def __init__(self, params=None, **kwargs):
        super(NormalizedHF, self).__init__(params, **kwargs)

    @classmethod
    def algorithm(cls, data, params):
        return 1 - NormalizedLF.get(data, params)

    @classmethod
    def get_used_params(cls):
        return NormalizedLF.get_used_params()


from scipy.spatial.distance import cdist as _cd, pdist as _pd
from scipy.stats.mstats import mquantiles as _mq


class ApproxEntropy(_Indicator):
    """
    Calculates the approx entropy of the data series.
    """

    def __init__(self, params=None, **kwargs):
        super(ApproxEntropy, self).__init__(params, **kwargs)

    @classmethod
    def algorithm(cls, data, params):
        assert 'approx_entropy_r' in params, "This feature needs the parameter 'approx_entropy_r'."
        if len(data) < 3:
            return _np.nan
        else:
            r = params['approx_entropy_r']
            uj_m = OrderedSubsets.get(data, subset_size=2)
            uj_m1 = OrderedSubsets.get(data, subset_size=3)
            card_elem_m = uj_m.shape[0]
            card_elem_m1 = uj_m1.shape[0]

            r = r * _np.std(data)
            d_m = _cd(uj_m, uj_m, 'chebyshev')
            d_m1 = _cd(uj_m1, uj_m1, 'chebyshev')

            cmr_m_ap_en = _np.zeros(card_elem_m)
            for i in xrange(card_elem_m):
                vector = d_m[i]
                cmr_m_ap_en[i] = float(sum(1 for i in vector if i <= r)) / card_elem_m

            cmr_m1_ap_en = _np.zeros(card_elem_m1)
            for i in xrange(card_elem_m1):
                vector = d_m1[i]
                cmr_m1_ap_en[i] = float(sum(1 for i in vector if i <= r)) / card_elem_m1

            phi_m = _np.sum(_np.log(cmr_m_ap_en)) / card_elem_m
            phi_m1 = _np.sum(_np.log(cmr_m1_ap_en)) / card_elem_m1

            return phi_m - phi_m1


class SampleEntropy(_Indicator):
    """
    Calculates the sample entropy of the data series.
    """

    def __init__(self, params=None, **kwargs):
        super(SampleEntropy, self).__init__(params, **kwargs)

    @classmethod
    def algorithm(cls, data, params):
        assert 'sample_entropy_r' in params, "This feature needs the parameter 'sample_entropy_r'."
        if len(data) < 4:
            return _np.nan
        else:
            r = params['sample_entropy_r']
            uj_m = OrderedSubsets.get(data, subset_size=2)
            uj_m1 = OrderedSubsets.get(data, subset_size=3)

            num_elem_m = uj_m.shape[0]
            num_elem_m1 = uj_m1.shape[0]

            r = r * SD.get(data)
            d_m = _cd(uj_m, uj_m, 'che' + 'bys' + 'hev')
            d_m1 = _cd(uj_m1, uj_m1, 'che' + 'bys' + 'hev')

            cmr_m_sa_mp_en = _np.zeros(num_elem_m)
            for i in xrange(num_elem_m):
                vector = d_m[i]
                cmr_m_sa_mp_en[i] = (sum(1 for i in vector if i <= r) - 1) / (num_elem_m - 1)

            cmr_m1_sa_mp_en = _np.zeros(num_elem_m1)
            for i in xrange(num_elem_m1):
                vector = d_m1[i]
                cmr_m1_sa_mp_en[i] = (sum(1 for i in vector if i <= r) - 1) / (num_elem_m1 - 1)

            cm = _np.sum(cmr_m_sa_mp_en) / num_elem_m
            cm1 = _np.sum(cmr_m1_sa_mp_en) / num_elem_m1

            return _np.log(cm / cm1)


class FractalDimension(_Indicator):
    """
    Calculates the fractal dimension of the data series.
    """

    def __init__(self, params=None, **kwargs):
        super(FractalDimension, self).__init__(params, **kwargs)

    @classmethod
    def algorithm(cls, data, params):
        assert 'cra' in params, "This feature needs the parameter 'cra'."
        assert 'crb' in params, "This feature needs the parameter 'crb'."
        if len(data) < 3:
            return _np.nan
        else:
            uj_m = OrderedSubsets.get(data, subset_size=2)
            cra = params['cra']
            crb = params['crb']
            mutual_distance = _pd(uj_m, 'che' + 'bys' + 'hev')

            num_elem = len(mutual_distance)

            rr = _mq(mutual_distance, prob=[cra, crb])
            ra = rr[0]
            rb = rr[1]

            cmr_a = (sum(1 for i in mutual_distance if i <= ra)) / num_elem
            cmr_b = (sum(1 for i in mutual_distance if i <= rb)) / num_elem

            return (_np.log(cmr_b) - _np.log(cmr_a)) / (_np.log(rb) - _np.log(ra))


class SVDEntropy(_Indicator):
    """
    Calculates the SVD entropy of the data series.
    """

    def __init__(self, params=None, **kwargs):
        super(SVDEntropy, self).__init__(params, **kwargs)

    @classmethod
    def algorithm(cls, data, params):
        if len(data) < 2:
            return _np.nan
        else:
            uj_m = OrderedSubsets.get(data, subset_size=2)
            w = _np.linalg.svd(uj_m, compute_uv=False)
            w /= sum(w)
            return -1 * sum(w * _np.log(w))


class Fisher(_Indicator):
    """
    Calculates the Fisher index of the data series.
    """

    def __init__(self, params=None, **kwargs):
        super(Fisher, self).__init__(params, **kwargs)

    @classmethod
    def algorithm(cls, data, params):
        if len(data) < 2:
            return _np.nan
        else:
            uj_m = OrderedSubsets.get(data, subset_size=2)
            w = _np.linalg.svd(uj_m, compute_uv=False)
            w /= sum(w)
            fi = 0
            for i in xrange(0, len(w) - 1):  # from Test1 to M
                fi += ((w[i + 1] - w[i]) ** 2) / (w[i])

            return fi


class CorrelationDim(_Indicator):
    """
    Calculates the correlation dimension of the data series.
    """

    def __init__(self, params=None, **kwargs):
        super(CorrelationDim, self).__init__(params, **kwargs)

    @classmethod
    def algorithm(cls, data, params):
        assert 'corr_dim_len' in params, "This feature needs the parameter 'corr_dim_len'."
        if len(data) < params['corr_dim_len']:
            return _np.nan
        else:
            rr = data  # rr in seconds
            # Check also the other indicators to work with seconds!
            uj = OrderedSubsets.get(rr, dict(subset_size=params['corr_dim_len']))
            num_elem = uj.shape[0]
            r_vector = _np.arange(0.3, 0.46, 0.02)  # settings
            c = _np.zeros(len(r_vector))
            jj = 0
            n = _np.zeros(num_elem)
            dj = _cd(uj, uj)
            for r in r_vector:
                for i in xrange(num_elem):
                    vector = dj[i]
                    n[i] = float(sum(1 for i in vector if i <= r)) / num_elem
                c[jj] = _np.sum(n) / num_elem
                jj += 1

            log_c = _np.log(c)
            log_r = _np.log(r_vector)

            return (log_c[-1] - log_c[0]) / (log_r[-1] - log_r[0])


class PoinSD1(_Indicator):
    """
    Calculates the SD1 Poincaré index of the data series.
    """

    def __init__(self, params=None, **kwargs):
        super(PoinSD1, self).__init__(params, **kwargs)

    @classmethod
    def algorithm(cls, data, params):
        sd1, sd2 = PoincareSD.get(data)
        return sd1


class PoinSD2(_Indicator):
    """
    Calculates the SD2 Poincaré index of the data series.
    """

    def __init__(self, params=None, **kwargs):
        super(PoinSD2, self).__init__(params, **kwargs)

    @classmethod
    def algorithm(cls, data, params):
        sd1, sd2 = PoincareSD.get(data)
        return sd2


class PoinSD12(_Indicator):
    """
    Calculates the ratio between SD1 and SD2 Poincaré indicators of the data series.
    """

    def __init__(self, params=None, **kwargs):
        super(PoinSD12, self).__init__(params, **kwargs)

    @classmethod
    def algorithm(cls, data, params):
        sd1, sd2 = PoincareSD.get(data)
        return sd1 / sd2


class PoinEll(_Indicator):
    """
    Calculates the Poincaré Ell. index of the data series.
    """

    def __init__(self, params=None, **kwargs):
        super(PoinEll, self).__init__(params, **kwargs)

    @classmethod
    def algorithm(cls, data, params):
        sd1, sd2 = PoincareSD.get(data)
        return sd1 * sd2 * _np.pi


class Hurst(_Indicator):
    """
    Calculates the Hurst HRV index of the data series.
    """

    def __init__(self, params=None, **kwargs):
        super(Hurst, self).__init__(params, **kwargs)

    @classmethod
    def algorithm(cls, data, params):
        n = len(data)
        if n < 2:
            return _np.nan
        else:
            t = _np.arange(1.0, n + 1)
            y = _np.cumsum(data)
            ave_t = _np.array(y / t)

            s_t = _np.zeros(n)
            r_t = _np.zeros(n)
            for i in xrange(n):
                s_t[i] = _np.std(data[:i + 1])
                x_t = y - t * ave_t[i]
                r_t[i] = _np.max(x_t[:i + 1]) - _np.min(x_t[:i + 1])

            r_s = r_t / s_t
            r_s = _np.log(r_s)
            n = _np.log(t).reshape(n, 1)
            h = _np.linalg.lstsq(n[1:], r_s[1:])[0]
            return h[0]


class PetrosianFracDim(_Indicator):
    """
    Calculates the petrosian's fractal dimension of the data series.
    """

    def __init__(self, params=None, **kwargs):
        super(PetrosianFracDim, self).__init__(params, **kwargs)

    @classmethod
    def algorithm(cls, data, params):
        d = _Diff.get(data)
        n_delta = 0  # number of sign changes in derivative of the signal
        for i in xrange(1, len(d)):
            if d[i] * d[i - 1] < 0:
                n_delta += 1
        n = len(data)
        return _np.float(_np.log10(n) / (_np.log10(n) + _np.log10(n / n + 0.4 * n_delta)))


class DFAShortTerm(_Indicator):
    """
    Calculate the alpha1 (short term) component index of the De-trended Fluctuation Analysis.
    """

    def __init__(self, params=None, **kwargs):
        super(DFAShortTerm, self).__init__(params, **kwargs)

    @classmethod
    def algorithm(cls, data, params):
        # calculates De-trended Fluctuation Analysis: alpha1 (short term) component
        x = data
        if len(x) < 16:
            return _np.nan
        else:
            ave = Mean.get(x)
            y = _np.cumsum(x)
            y -= ave
            l = _np.arange(4, 17, 4)
            f = _np.zeros(len(l))  # f(n) of different given box length n
            for i in xrange(0, len(l)):
                n = int(l[i])  # for each box length l[i]
                for j in xrange(0, len(x), n):  # for each box
                    if j + n < len(x):
                        c = range(j, j + n)
                        c = _np.vstack([c, _np.ones(n)]).T  # coordinates of time in the box
                        z = y[j:j + n]  # the value of example_data in the box
                        f[i] += _np.linalg.lstsq(c, z)[1]  # add residue in this box
                f[i] /= ((len(x) / n) * n)
            f = _np.sqrt(f)
            return _np.linalg.lstsq(_np.vstack([_np.log(l), _np.ones(len(l))]).T, _np.log(f))[0][0]


class DFALongTerm(_Indicator):
    """
    Calculate the alpha2 (long term) component index of the De-trended Fluctuation Analysis.
    """

    def __init__(self, params=None, **kwargs):
        super(DFALongTerm, self).__init__(params, **kwargs)

    @classmethod
    def algorithm(cls, data, params):
        # calculates De-trended Fluctuation Analysis: alpha2 (long term) component
        x = data
        if len(x) < 16:
            return _np.nan
        else:
            ave = Mean.get(x)
            y = _np.cumsum(x)
            y -= ave
            l_max = _np.min([64, len(x)])
            l = _np.arange(16, l_max + 1, 4)
            f = _np.zeros(len(l))  # f(n) of different given box length n
            for i in xrange(0, len(l)):
                n = int(l[i])  # for each box length l[i]
                for j in xrange(0, len(x), n):  # for each box
                    if j + n < len(x):
                        c = range(j, j + n)
                        c = _np.vstack([c, _np.ones(n)]).T  # coordinates of time in the box
                        z = y[j:j + n]  # the value of example_data in the box
                        f[i] += _np.linalg.lstsq(c, z)[1]  # add residue in this box
                f[i] /= ((len(x) / n) * n)
            f = _np.sqrt(f)

            return _np.linalg.lstsq(_np.vstack([_np.log(l), _np.ones(len(l))]).T, _np.log(f))[0][0]
