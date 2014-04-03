__author__ = 'AleB'

__all__ = ['HF', 'HFNormal', 'HFPeak', 'LF', 'LFHF', 'LFNormal', 'LFPeak', 'NormalHF', 'NormalLF', 'Total', 'VLF',
           'VLFNormal', 'VLFPeak']

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

        indexes = np.array([i for i in range(len(spec)) if freq_min <= freq[i] < freq_max])
        self._freq_band = freq[indexes]
        self._spec_band = spec[indexes]
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
    def __init__(self, data=None):
        super(VLF, self).__init__(Sett.StandardBands.vlf_lower_bound, Sett.StandardBands.vlf_upper_bound, data)
        self._value = VLF._calculate_data(self)


class LF(PowerInBand):
    def __init__(self, data=None):
        super(LF, self).__init__(Sett.StandardBands.vlf_upper_bound, Sett.StandardBands.lf_upper_bound, data)
        # .get(..) called on LF only for the .cid() in the cache. The actually important data is self._freq_band that
        # has been calculated by PowerInBand.__init__(..)
        self._value = LF.get(self)


class HF(PowerInBand):
    def __init__(self, data=None):
        super(HF, self).__init__(Sett.StandardBands.lf_upper_bound, Sett.StandardBands.hf_upper_bound, data)
        # Here as in LF
        self._value = HF.get(self)


class Total(PowerInBand):
    def __init__(self, data=None):
        super(Total, self).__init__(Sett.StandardBands.vlf_lower_bound, Sett.StandardBands.lf_upper_bound, data)
        # Used _calculate_data(..) (here as in other indexes) as a substitute of the ex 'calculate' to bypass the
        # cache system
        self._value = Total._calculate_data(self)


class VLFPeak(PeakInBand):
    def __init__(self, data=None):
        super(VLFPeak, self).__init__(Sett.StandardBands.vlf_lower_bound, Sett.StandardBands.vlf_upper_bound, data)


class LFPeak(PeakInBand):
    def __init__(self, data=None):
        super(LFPeak, self).__init__(Sett.StandardBands.vlf_upper_bound, Sett.StandardBands.lf_upper_bound, data)


class HFPeak(PeakInBand):
    def __init__(self, data=None):
        super(HFPeak, self).__init__(Sett.StandardBands.lf_upper_bound, Sett.StandardBands.hf_upper_bound, data)


class VLFNormal(PowerInBandNormal):
    def __init__(self, data=None):
        super(VLFNormal, self).__init__(Sett.StandardBands.vlf_lower_bound, Sett.StandardBands.vlf_upper_bound, data)


class LFNormal(PowerInBandNormal):
    def __init__(self, data=None):
        super(LFNormal, self).__init__(Sett.StandardBands.vlf_upper_bound, Sett.StandardBands.lf_upper_bound, data)


class HFNormal(PowerInBandNormal):
    def __init__(self, data=None):
        super(HFNormal, self).__init__(Sett.StandardBands.lf_upper_bound, Sett.StandardBands.hf_upper_bound, data)


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
