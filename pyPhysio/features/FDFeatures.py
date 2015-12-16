from __future__ import division

__author__ = 'AleB'
__all__ = ['PowerInBand', 'PeakInBand', 'PowerInBandNormal', 'LFHF', 'NormalizedHF', 'NormalizedLF']

from numpy import argmax, sum
from pyPhysio.BaseFeature import Feature
from CacheOnlyFeatures import PSDWelchLibCalc


class FDFeature(Feature):
    """
    This is the base class for the Frequency Domain Features.
    It uses the interpolation frequency parameter interp_freq.
    """

    def __init__(self, params=None, _kwargs=None):
        super(FDFeature, self).__init__(params, _kwargs)
        assert 'interp_freq' in self._params, "This feature needs 'interp_freq'."
        self._interp_freq = self._params['interp_freq']

    @classmethod
    def algorithm(cls, data, params):
        """
        Placeholder for the subclasses
        @raise NotImplementedError: Ever
        """
        raise NotImplementedError(cls.__name__ + " is an FDFeature but it is not implemented.")

    @classmethod
    def get_used_params(cls):
        return ['interp_freq', 'psd_method']


class InBand(FDFeature):

    def __init__(self, params=None, **kwargs):
        super(InBand, self).__init__(params, kwargs)

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


class PowerInBand(FDFeature):

    def __init__(self, params=None, **kwargs):
        super(PowerInBand, self).__init__(params, kwargs)

    @classmethod
    def algorithm(cls, data, params):
        ignore, _pow_band, ignored = InBand.get(data, params)
        return sum(_pow_band) / len(_pow_band)

    @classmethod
    def get_used_params(cls):
        return InBand.get_used_params()


class PowerInBandNormal(FDFeature):

    def __init__(self, params=None, **kwargs):
        super(PowerInBandNormal, self).__init__(params, kwargs)

    @classmethod
    def algorithm(cls, data, params):
        ignored, _pow_band, _pow_total = InBand.get(data, params)
        return sum(_pow_band) / len(_pow_band) / _pow_total

    @classmethod
    def get_used_params(cls):
        return InBand.get_used_params()


class PeakInBand(FDFeature):

    def __init__(self, params=None, **kwargs):
        super(PeakInBand, self).__init__(params, kwargs)

    @classmethod
    def algorithm(cls, data, params):
        _freq_band, _pow_band, ignored = InBand.get(data, params)
        return _freq_band[argmax(_pow_band)]

    @classmethod
    def get_used_params(cls):
        return InBand.get_used_params()


class LFHF(FDFeature):

    def __init__(self, params=None, **kwargs):
        super(LFHF, self).__init__(params, kwargs)

    @classmethod
    def algorithm(cls, data, params):
        assert 'freq_mid' in params, "Need the parameter 'freq_mid' as the separator between LF and HF."
        par_lf = params.copy().update({'freq_max': params['freq_mid']})
        par_hf = params.copy().update({'freq_min': params['freq_mid']})
        return PowerInBand.get(data, par_lf) / PowerInBand.get(data, par_hf)

    @classmethod
    def get_used_params(cls):
        return PowerInBand.get_used_params()


class NormalizedLF(FDFeature):
    """
    Calculates the normalized power value of the LF band (parametrized in the settings) over the LF and HF bands.
    """

    def __init__(self, params=None, **kwargs):
        super(NormalizedLF, self).__init__(params, kwargs)

    @classmethod
    def algorithm(cls, data, params):
        assert 'freq_mid' in params, "Need the parameter 'freq_mid' as the separator between LF and HF."
        par_lf = params.copy().update({'freq_max': params['freq_mid']})
        par_hf = params.copy().update({'freq_min': params['freq_mid']})
        return PowerInBand.get(data, par_lf) / (PowerInBand.get(data, par_hf) + PowerInBand.get(data, par_lf))

    @classmethod
    def get_used_params(cls):
        return PowerInBand.get_used_params() + ['freq_mid']


class NormalizedHF(FDFeature):
    """
    Calculates the normalized power value of the HF band (parametrized in the settings) over the LF and HF bands.
    """

    def __init__(self, params=None, **kwargs):
        super(NormalizedHF, self).__init__(params, kwargs)

    @classmethod
    def algorithm(cls, data, params):
        return 1 - NormalizedLF.get(data, params)

    @classmethod
    def get_used_params(cls):
        return NormalizedLF.get_used_params()
