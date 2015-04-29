from __future__ import division

__author__ = 'AleB'
__all__ = ['PowerInBand', 'PowerInBandNormal', 'LFHF', 'NormalizedHF', 'NormalizedLF']

from numpy import argmax, sum
from pyPhysio.indexes.BaseFeatures import FDFeature
from pyPhysio.PyHRVSettings import MainSettings as Sett


class InBand(FDFeature):
    def __init__(self, data, params):
        super(InBand, self).__init__(data, params)
        self._freq_band, self._spec_band = InBand.get(data, params)
        ignored, ignored, self._total_band = Sett.psd_algorithm.get(data, params)
        self._value = self._total_band

    @classmethod
    def _compute(cls, data, params):
        assert params is not None
        assert 'freq_min' in params
        assert 'freq_max' in params

        freq, spec, total = Sett.psd_algorithm.get(data, params)

        return ([freq[i] for i in xrange(len(freq)) if params['freq_min'] <= freq[i] < params['freq_max']],
                [spec[i] for i in xrange(len(spec)) if params['freq_min'] <= freq[i] < params['freq_max']])

    @staticmethod
    def get_used_params():
        return ['freq_max', 'freq_min'].extend(Sett.psd_algorithm.get_used_params())


class PowerInBand(InBand):
    def __init__(self, data, params):
        super(PowerInBand, self).__init__(data, params)
        self._power_in_band = self._value = sum(self._spec_band) / len(self._freq_band)


class PowerInBandNormal(PowerInBand):
    def __init__(self, data, params):
        super(PowerInBandNormal, self).__init__(data, params)
        # Ok if cache used, else recalculate
        self._value = self._power_in_band / self._total_band


class PeakInBand(InBand):
    def __init__(self, data, params):
        super(PeakInBand, self).__init__(data, params)
        self._value = self._freq_band[argmax(self._spec_band)]


class LFHF(FDFeature):
    """
    Calculates the power ratio between the LF and the HF band (parametrized in the settings).
    """

    def __init__(self, data, params):
        """
        Calculates the power ratio between the LF and the HF band (parametrized in the settings).
        """
        super(FDFeature, self).__init__(data, params)
        par_lf = params.copy().update({'freq_max': params['mid_freq']})
        par_hf = params.copy().update({'freq_min': params['mid_freq']})
        self._value = PowerInBand(self._data, par_lf).value / PowerInBand(self._data, par_hf).value


class NormalizedLF(FDFeature):
    """
    Calculates the normalized power value of the LF band (parametrized in the settings) over the LF and HF bands.
    """

    def __init__(self, data, params):
        """
        Calculates the normalized power value of the LF band (parametrized in the settings) over the LF and HF bands.
        """
        super(FDFeature, self).__init__(data, params)
        par_lf = params.copy().update({'freq_max': params['mid_freq']})
        par_hf = params.copy().update({'freq_min': params['mid_freq']})
        self._value = PowerInBand(self._data, par_lf).value / \
            (PowerInBand(self._data, par_hf).value + PowerInBand(self._data, par_lf).value)


class NormalizedHF(FDFeature):
    """
    Calculates the normalized power value of the HF band (parametrized in the settings) over the LF and HF bands.
    """

    def __init__(self, data, params):
        """
        Calculates the normalized power value of the HF band (parametrized in the settings) over the LF and HF bands.
        """
        super(FDFeature, self).__init__(data, params)
        par_lf = params.copy().update({'freq_max': params['mid_freq']})
        par_hf = params.copy().update({'freq_min': params['mid_freq']})
        self._value = PowerInBand(self._data, par_hf).value / \
            (PowerInBand(self._data, par_hf).value + PowerInBand(self._data, par_lf).value)
