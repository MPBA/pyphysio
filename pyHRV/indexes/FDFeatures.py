from __future__ import division

__author__ = 'AleB'
__all__ = ['HF', 'HFNormal', 'HFPeak', 'LF', 'LFHF', 'LFNormal', 'LFPeak', 'NormalizedHF', 'NormalizedLF', 'Total',
           'VLF', 'VLFNormal', 'VLFPeak']

import numpy as np
from pyHRV.indexes.BaseFeatures import FDFeature
from pyHRV.PyHRVSettings import MainSettings as Sett


class InBand(FDFeature):
    def __init__(self, data, params):
        super(InBand, self).__init__(data, params)

        assert params is not None
        assert 'freq_min' in params
        assert 'freq_max' in params

        freq, spec, total = Sett.psd_algorithm.get(self._data, {'interp_freq': self._interp_freq})

        self._freq_band = [freq[i] for i in xrange(len(freq)) if params['freq_min'] <= freq[i] < params['freq_max']]
        self._spec_band = [spec[i] for i in xrange(len(spec)) if params['freq_min'] <= freq[i] < params['freq_max']]
        self._total_band = total


class PowerInBand(InBand):
    def __init__(self, data, params=None):
        super(PowerInBand, self).__init__(data, params)

    @classmethod
    def _compute(cls, self, params):
        """
        Calculates the data to cache using an InBand FDFeature instance as data.
        @param self: The InBand FDFeature instance as data.
        @param params: Unused
        @return: Power in the band
        @rtype: float
        """
        return np.sum(self._spec_band) / len(self._freq_band)


class PowerInBandNormal(InBand):
    def __init__(self, data, params):
        super(PowerInBandNormal, self).__init__(data, params)
        self._value = (np.sum(self._spec_band) / len(self._freq_band)) / self._total_band


class PeakInBand(InBand):
    def __init__(self, data, params):
        super(PeakInBand, self).__init__(data, params)
        self._value = self._freq_band[np.argmax(self._spec_band)]


class Total(PowerInBand):
    """
    Calculates the power of the whole spectrum.
    """

    def __init__(self, data):
        """
        Calculates the power of the whole spectrum.
        """
        super(Total, self).__init__(Sett.vlf_band_lower_bound, Sett.lf_band_upper_bound, data) # TODO: WAT? Total = vlf-lf?
        # Used _calculate_data(..) (here as in other indexes) as a substitute of the ex 'calculate' to bypass the
        # cache system
        self._value = Total._compute(self, {})


class VLFPeak(PeakInBand):
    """
    Calculates the peak in the VLF band (parametrized in the settings).
    """

    def __init__(self, data):
        """
        Calculates the peak in the VLF band (parametrized in the settings).
        """
        super(VLFPeak, self).__init__(Sett.vlf_band_lower_bound, Sett.vlf_band_upper_bound, data)


class LFPeak(PeakInBand):
    """
    Calculates the peak in the LF band (parametrized in the settings).
    """

    def __init__(self, data):
        """
        Calculates the peak in the LF band (parametrized in the settings).
        """
        super(LFPeak, self).__init__(Sett.vlf_band_upper_bound, Sett.lf_band_upper_bound, data)


class HFPeak(PeakInBand):
    """
    Calculates the peak in the HF band (parametrized in the settings).
    """

    def __init__(self, data):
        """
        Calculates the peak in the HF band (parametrized in the settings).
        """
        super(HFPeak, self).__init__(Sett.lf_band_upper_bound, Sett.hf_band_upper_bound, data)


class VLFNormal(PowerInBandNormal):
    """
    Calculates the normal of the VLF band (parametrized in the settings).
    """

    def __init__(self, data):
        """
        Calculates the normal of the VLF band (parametrized in the settings).
        """
        super(VLFNormal, self).__init__(Sett.vlf_band_lower_bound, Sett.vlf_band_upper_bound, data)


class LFNormal(PowerInBandNormal):
    """
    Calculates the normal of the LF band (parametrized in the settings).
    """

    def __init__(self, data):
        """
        Calculates the normal of the LF band (parametrized in the settings).
        """
        super(LFNormal, self).__init__(Sett.vlf_band_upper_bound, Sett.lf_band_upper_bound, data)


class HFNormal(PowerInBandNormal):
    """
    Calculates the normal of the HF band (parametrized in the settings).
    """

    def __init__(self, data):
        """
        Calculates the normal of the HF band (parametrized in the settings).
        """
        super(HFNormal, self).__init__(Sett.lf_band_upper_bound, Sett.hf_band_upper_bound, data)


class LFHF(FDFeature):
    """
    Calculates the power ratio between the LF and the HF band (parametrized in the settings).
    """

    def __init__(self, data):
        """
        Calculates the power ratio between the LF and the HF band (parametrized in the settings).
        """
        super(FDFeature, self).__init__(data)
        self._value = LF(self._data).value / HF(self._data).value


class NormalizedLF(FDFeature):
    """
    Calculates the normalized power value of the LF band (parametrized in the settings) over the LF and HF bands.
    """

    def __init__(self, data):
        """
        Calculates the normalized power value of the LF band (parametrized in the settings) over the LF and HF bands.
        """
        super(FDFeature, self).__init__(data)
        self._value = LF(self._data).value / (HF(self._data).value + LF(self._data).value)


class NormalizedHF(FDFeature):
    """
    Calculates the normalized power value of the HF band (parametrized in the settings) over the LF and HF bands.
    """

    def __init__(self, data):
        """
        Calculates the normalized power value of the HF band (parametrized in the settings) over the LF and HF bands.
        """
        super(FDFeature, self).__init__(data)
        self._value = HF(self._data).value / (HF(self._data).value + LF(self._data).value)
