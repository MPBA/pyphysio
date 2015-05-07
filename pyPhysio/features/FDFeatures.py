from __future__ import division

__author__ = 'AleB'
__all__ = ['PowerInBand', 'PeakInBand', 'PowerInBandNormal', 'LFHF', 'NormalizedHF', 'NormalizedLF']

from numpy import argmax, sum
from pyPhysio.features.BaseFeatures import FDFeature
from CacheOnlyFeatures import PSDWelchLibCalc


class InBand(FDFeature):

    def __init__(self, params=None, **kwargs):
        super(InBand, self).__init__(params, kwargs)

    @classmethod
    def raw_compute(cls, data, params):
        assert 'freq_min' in params, "Need the parameter 'freq_min' as the lower bound of the band."
        assert 'freq_max' in params, "Need the parameter 'freq_max' as the higher bound of the band."

        if 'psd_method' not in params:
            params.update({'psd_method': PSDWelchLibCalc})

        freq, spec, total = params['psd_method'].get(data, params)

        return ([freq[i] for i in xrange(len(freq)) if params['freq_min'] <= freq[i] < params['freq_max']],
                [spec[i] for i in xrange(len(spec)) if params['freq_min'] <= freq[i] < params['freq_max']],
                total)

    @staticmethod
    def get_used_params():
        return ['freq_max', 'freq_min'] + PSDWelchLibCalc.get_used_params()


class PowerInBand(FDFeature):

    def __init__(self, params=None, **kwargs):
        super(PowerInBand, self).__init__(params, kwargs)

    @classmethod
    def raw_compute(cls, data, params):
        ignore, _pow_band, ignored = InBand.get(data, params)
        return sum(_pow_band) / len(_pow_band)

    @staticmethod
    def get_used_params():
        return InBand.get_used_params()


class PowerInBandNormal(FDFeature):

    def __init__(self, params=None, **kwargs):
        super(PowerInBandNormal, self).__init__(params, kwargs)

    @classmethod
    def raw_compute(cls, data, params):
        ignored, _pow_band, _pow_total = InBand.get(data, params)
        return sum(_pow_band) / len(_pow_band) / _pow_total

    @staticmethod
    def get_used_params():
        return InBand.get_used_params()


class PeakInBand(FDFeature):

    def __init__(self, params=None, **kwargs):
        super(PeakInBand, self).__init__(params, kwargs)

    @classmethod
    def raw_compute(cls, data, params):
        _freq_band, _pow_band, ignored = InBand.get(data, params)
        return _freq_band[argmax(_pow_band)]

    @staticmethod
    def get_used_params():
        return InBand.get_used_params()


class LFHF(FDFeature):

    def __init__(self, params=None, **kwargs):
        super(LFHF, self).__init__(params, kwargs)

    @classmethod
    def raw_compute(cls, data, params):
        assert 'freq_mid' in params, "Need the parameter 'freq_mid' as the separator between LF and HF."
        par_lf = params.copy().update({'freq_max': params['freq_mid']})
        par_hf = params.copy().update({'freq_min': params['freq_mid']})
        return PowerInBand.get(data, par_lf) / PowerInBand.get(data, par_hf)

    @staticmethod
    def get_used_params():
        return PowerInBand.get_used_params()


class NormalizedLF(FDFeature):
    """
    Calculates the normalized power value of the LF band (parametrized in the settings) over the LF and HF bands.
    """

    def __init__(self, params=None, **kwargs):
        super(NormalizedLF, self).__init__(params, kwargs)

    @classmethod
    def raw_compute(cls, data, params):
        assert 'freq_mid' in params, "Need the parameter 'freq_mid' as the separator between LF and HF."
        par_lf = params.copy().update({'freq_max': params['freq_mid']})
        par_hf = params.copy().update({'freq_min': params['freq_mid']})
        return PowerInBand.get(data, par_lf) / (PowerInBand.get(data, par_hf) + PowerInBand.get(data, par_lf))

    @staticmethod
    def get_used_params():
        return PowerInBand.get_used_params() + ['freq_mid']


class NormalizedHF(FDFeature):
    """
    Calculates the normalized power value of the HF band (parametrized in the settings) over the LF and HF bands.
    """

    def __init__(self, params=None, **kwargs):
        super(NormalizedHF, self).__init__(params, kwargs)

    @classmethod
    def raw_compute(cls, data, params):
        return 1 - NormalizedLF.get(data, params)

    @staticmethod
    def get_used_params():
        return NormalizedLF.get_used_params()
