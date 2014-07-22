##ck3
__author__ = 'AleB'
__all__ = ['HF', 'HFNormal', 'HFPeak', 'LF', 'LFHF', 'LFNormal', 'LFPeak', 'NormalizedHF', 'NormalizedLF', 'Total',
           'VLF', 'VLFNormal', 'VLFPeak']

import numpy as np

from pyHRV.indexes.BaseIndexes import FDIndex
from pyHRV.Cache import CacheableDataCalc, PSDWelchCalc
from pyHRV.PyHRVSettings import PyHRVDefaultSettings as Sett


class InBand(FDIndex):
    def __init__(self, freq_min, freq_max, interp_freq=Sett.default_interpolation_freq, data=None):
        super(InBand, self).__init__(interp_freq, data)
        self._freq_min = freq_min
        self._freq_max = freq_max

        freq, spec, total = PSDWelchCalc.get(self._data, self._interp_freq)

        self._freq_band = [freq[i] for i in xrange(len(freq)) if freq_min <= freq[i] < freq_max]
        self._spec_band = [spec[i] for i in xrange(len(spec)) if freq_min <= freq[i] < freq_max]
        self._total_band = total


class PowerInBand(InBand, CacheableDataCalc):
    def __init__(self, freq_min, freq_max, data=None, interp_freq=Sett.default_interpolation_freq):
        super(PowerInBand, self).__init__(freq_min, freq_max, interp_freq, data)

    @classmethod
    def _calculate_data(cls, self, params=None):
        return np.sum(self._spec_band) / len(self._freq_band)


class PowerInBandNormal(InBand):
    def __init__(self, freq_min, freq_max, data=None, interp_freq=Sett.default_interpolation_freq):
        super(PowerInBandNormal, self).__init__(freq_min, freq_max, interp_freq, data)
        self._value = (np.sum(self._spec_band) / len(self._freq_band)) / self._total_band


class PeakInBand(InBand):
    def __init__(self, freq_min, freq_max, data=None, interp_freq=Sett.default_interpolation_freq):
        super(PeakInBand, self).__init__(freq_min, freq_max, interp_freq, data=data)
        self._value = self._freq_band[np.argmax(self._spec_band)]


class VLF(PowerInBand):
    """
    Calculates the power in the VLF band (parametrized in the settings).
    """

    def __init__(self, data):
        """
        Calculates the power in the VLF band (parametrized in the settings).
        """
        super(VLF, self).__init__(Sett.vlf_band_lower_bound, Sett.vlf_band_upper_bound, data)
        self._value = VLF.get(self)


class LF(PowerInBand):
    """
    Calculates the power in the LF band (parametrized in the settings).
    """

    def __init__(self, data):
        """
        Calculates the power in the LF band (parametrized in the settings).
        """
        super(LF, self).__init__(Sett.vlf_band_upper_bound, Sett.lf_band_upper_bound, data)
        # .get(..) called on LF only for the .cid() in the cache. The actually important data is self._freq_band that
        # has been calculated by PowerInBand.__init__(..)
        self._value = LF.get(self)


class HF(PowerInBand):
    """
    Calculates the power in the HF band (parametrized in the settings).
    """

    def __init__(self, data):
        """
        Calculates the power in the HF band (parametrized in the settings).
        """
        super(HF, self).__init__(Sett.lf_band_upper_bound, Sett.hf_band_upper_bound, data)
        # Here as in LF
        self._value = HF.get(self)


class Total(PowerInBand):
    """
    Calculates the power of the whole spectrum.
    """

    def __init__(self, data):
        """
        Calculates the power of the whole spectrum.
        """
        super(Total, self).__init__(Sett.vlf_band_lower_bound, Sett.lf_band_upper_bound, data)
        # Used _calculate_data(..) (here as in other indexes) as a substitute of the ex 'calculate' to bypass the
        # cache system
        self._value = Total._calculate_data(self)


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


class LFHF(FDIndex):
    """
    Calculates the power ratio between the LF and the HF band (parametrized in the settings).
    """

    def __init__(self, data):
        """
        Calculates the power ratio between the LF and the HF band (parametrized in the settings).
        """
        super(FDIndex, self).__init__(data)
        self._value = LF(self._data).value / HF(self._data).value


class NormalizedLF(FDIndex):
    """
    Calculates the normalized power value of the LF band (parametrized in the settings) over the LF and HF bands.
    """

    def __init__(self, data):
        """
        Calculates the normalized power value of the LF band (parametrized in the settings) over the LF and HF bands.
        """
        super(FDIndex, self).__init__(data)
        self._value = LF(self._data).value / (HF(self._data).value + LF(self._data).value)


class NormalizedHF(FDIndex):
    """
    Calculates the normalized power value of the HF band (parametrized in the settings) over the LF and HF bands.
    """

    def __init__(self, data):
        """
        Calculates the normalized power value of the HF band (parametrized in the settings) over the LF and HF bands.
        """
        super(FDIndex, self).__init__(data)
        self._value = HF(self._data).value / (HF(self._data).value + LF(self._data).value)
