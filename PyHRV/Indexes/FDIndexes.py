__author__ = 'ale'

import numpy as np
from PyHRV.Indexes.Indexes import FDIndex
from PyHRV.DataSeries import CacheableDataCalc, PSDWelchCalc
from PyHRVSettings import PyHRVDefaultSettings as Sett


class InBand(FDIndex):
    def __init__(self, fmin, fmax, interp_freq=Sett.interpolation_freq_default, data=None, value=None):
        super(InBand, self).__init__(interp_freq, data, value)
        self._fmin = fmin
        self._fmax = fmax

        freq, spec, total = PSDWelchCalc.get(self._data, self._interp_freq)

        indexes = np.array([i for i in range(len(spec)) if fmin <= freq[i] < fmax])
        self._freq_band = freq[indexes]
        self._spec_band = spec[indexes]
        self._total_band = total


class PowerInBand(InBand, CacheableDataCalc):
    def __init__(self, fmin, fmax, data=None, interp_freq=Sett.interpolation_freq_default):
        super(PowerInBand, self).__init__(fmin, fmax, interp_freq, data=data)

    @classmethod
    def _calculate_data(cls, self, params=None):
        return np.sum(self._spec_band) / len(self._freq_band)


class PowerInBandNormal(InBand):
    def __init__(self, fmin, fmax, data=None, interp_freq=Sett.interpolation_freq_default):
        super(PowerInBandNormal, self).__init__(fmin, fmax, interp_freq, data=data)
        self._value = (np.sum(self._spec_band) / len(self._freq_band)) / self._total_band


class PeakInBand(InBand):
    def __init__(self, fmin, fmax, data=None, interp_freq=Sett.interpolation_freq_default):
        super(PeakInBand, self).__init__(fmin, fmax, interp_freq, data=data)
        self._value = self._freq_band[np.argmax(self._spec_band)]


class VLF(PowerInBand):
    def __init__(self, data=None):
        super(VLF, self).__init__(Sett.bands_vlf_lower_bound, Sett.bands_vlf_upper_bound, data)
        self._value = VLF._calculate_data(self)


class LF(PowerInBand):
    def __init__(self, data=None):
        super(LF, self).__init__(Sett.bands_vlf_upper_bound, Sett.bands_lf_upper_bound, data)
        # .get(..) called on LF only for the .cid() in the cache. The actually important data is self._freq_band that
        # has been calculated by PowerInBand.__init__(..)
        self._value = LF.get(self)


class HF(PowerInBand):
    def __init__(self, data=None):
        super(HF, self).__init__(Sett.bands_lf_upper_bound, Sett.bands_hf_upper_bound, data)
        # Here as in LF
        self._value = HF.get(self)


class Total(PowerInBand):
    def __init__(self, data=None):
        super(Total, self).__init__(Sett.bands_vlf_lower_bound, Sett.bands_lf_upper_bound, data)
        # Used _calculate_data(..) (here as in other indexes) as a substitute of the ex 'calculate' to bypass the
        # cache system
        self._value = Total._calculate_data(self)


class VLFPeak(PeakInBand):
    def __init__(self, data=None):
        super(VLFPeak, self).__init__(Sett.bands_vlf_lower_bound, Sett.bands_vlf_upper_bound, data)


class LFPeak(PeakInBand):
    def __init__(self, data=None):
        super(LFPeak, self).__init__(Sett.bands_vlf_upper_bound, Sett.bands_lf_upper_bound, data)


class HFPeak(PeakInBand):
    def __init__(self, data=None):
        super(HFPeak, self).__init__(Sett.bands_lf_upper_bound, Sett.bands_hf_upper_bound, data)


class VLFNormal(PowerInBandNormal):
    def __init__(self, data=None):
        super(VLFNormal, self).__init__(Sett.bands_vlf_lower_bound, Sett.bands_vlf_upper_bound, data)


class LFNormal(PowerInBandNormal):
    def __init__(self, data=None):
        super(LFNormal, self).__init__(Sett.bands_vlf_upper_bound, Sett.bands_lf_upper_bound, data)


class HFNormal(PowerInBandNormal):
    def __init__(self, data=None):
        super(HFNormal, self).__init__(Sett.bands_lf_upper_bound, Sett.bands_hf_upper_bound, data)


class LFHF(FDIndex):
    def __init__(self, data=None):
        super(FDIndex, self).__init__(data)
        self._value = LF(self._data).value / HF(self._data).value


class NormalLF(FDIndex):
    def __init__(self, data=None):
        super(FDIndex, self).__init__(data)
        self._value = LF(self._data).value / (HF(self._data).value + LF(self._data).value)


class NormalHF(FDIndex):
    def __init__(self, data=None):
        super(FDIndex, self).__init__(data)
        self._value = HF(self._data).value / (HF(self._data).value + LF(self._data).value)
