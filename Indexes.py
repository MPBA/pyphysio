# coding=utf-8
__author__ = 'AleB'

from DataSeries import *
import numpy as np
from PyHRVSettings import PyHRVDefaultSettings as Sett


class DataAnalysis(object):
    pass


class Index(object):
    def __init__(self, data=None, value=None):
        self._value = value
        self._data = data

    @property
    def calculated(self):
        """
        Returns weather the index is already calculated and up-to-date
        @return: Boolean
        """
        return not (self._value is None)

    @property
    def value(self):
        return self._value

    def update(self, data):
        self._data = data
        self._value = None


class TDIndex(Index):
    def __init__(self, data=None, value=None):
        super(TDIndex, self).__init__(data, value)


class FDIndex(Index):
    def __init__(self, interp_freq, data=None, value=None):
        super(FDIndex, self).__init__(data, value)
        self._interp_freq = interp_freq


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
        """
        Spec-Freq
        @param spect_band:
        @param freq_band:
        @return:
        """
        return np.sum(self._spec_band)/len(self._freq_band)


class PowerInBandNormal(InBand):
    def __init__(self, fmin, fmax, data=None, interp_freq=Sett.interpolation_freq_default):
        super(PowerInBandNormal, self).__init__(fmin, fmax, interp_freq, data=data)
        self._value = (np.sum(self._spec_band)/len(self._freq_band))/self._total_band


class PeakInBand(InBand):
    def __init__(self, fmin, fmax, data=None, interp_freq=Sett.interpolation_freq_default):
        super(PeakInBand, self).__init__(fmin, fmax, interp_freq, data=data)
        self._value = self._freq_band[np.argmax(self._spec_band)]


#############
# TIME DOMAIN
#############
class RRMean(TDIndex, CacheableDataCalc):
    def __init__(self, data=None):
        super(RRMean, self).__init__(data)
        self._value = RRMean.get(self._data)

    @classmethod
    def _calculate_data(cls, data, params):
        return np.mean(data)


class HRMean(TDIndex, CacheableDataCalc):
    def __init__(self, data=None):
        super(HRMean, self).__init__(data)
        self._value = HRMean.get(self._data)

    @classmethod
    def _calculate_data(cls, data, params):
        return np.mean(60/data)


class RRMedian(TDIndex, CacheableDataCalc):
    def __init__(self, data=None):
        super(RRMedian, self).__init__(data)
        self._value = RRMedian.get(self._data)

    @classmethod
    def _calculate_data(cls, data, params):
        return np.median(data)


class HRMedian(TDIndex):
    def __init__(self, data=None):
        super(HRMedian, self).__init__(data)
        self._value = 60 / RRMedian.get(self._data)


class RRSTD(TDIndex, CacheableDataCalc):
    def __init__(self, data=None):
        super(RRSTD, self).__init__(data)
        self._value = RRSTD.get(self._data)

    @classmethod
    def _calculate_data(cls, data, params):
        return np.std(data)


class HRSTD(TDIndex, CacheableDataCalc):
    def __init__(self, data=None):
        super(TDIndex, self).__init__(data)
        self._value = HRSTD.get(self._data)

    @classmethod
    def _calculate_data(cls, data, params):
        return np.std(60/data)

## TODO: self._value= NNx/len(diff)
class PNNx(TDIndex):
    def __init__(self, threshold, data=None):
        super(TDIndex, self).__init__(data)
        self._xth = threshold
        diff = RRDiff.get(self._data)
        self._value = 100.0 * sum(1 for x in diff if x > self._xth) / len(diff)


class NNx(TDIndex):
    def __init__(self, threshold, data=None):
        super(TDIndex, self).__init__(data)
        self._xth = threshold
        diff = RRDiff.get(self._data)
        self._value = 100.0 * sum(1 for x in diff if x > self._xth)


class RMSSD(TDIndex):
    def __init__(self, data=None):
        super(TDIndex, self).__init__(data)
        diff = RRDiff.get(self._data)
        self._value = np.sqrt(sum(diff**2)/(len(diff)-1))


class SDSD(TDIndex):
    def __init__(self, data=None):
        super(TDIndex, self).__init__(data)
        diff = RRDiff.get(self._data)
        self._value = np.std(diff)


#############
# FREQ DOMAIN
#############
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
        self._value = LF(self._data).value / (HF(self._data).value+LF(self._data).value)


class NormalHF(FDIndex):
    def __init__(self, data=None):
        super(FDIndex, self).__init__(data)
        self._value = HF(self._data).value / (HF(self._data).value+LF(self._data).value)


#############
# NLIN DOMAIN
#############
# TODO: non-lin indexes


class PoinIndex(Index):
    def __init__(self, data=None):
        super(PoinIndex, self).__init__(data)


class NLIndex(Index):
    def __init__(self, data=None):
        super(NLIndex, self).__init__(data)


class RRAnalysis(DataAnalysis):
    """ Static class containing methods for analyzing RR intervals data. """

    def __init__(self):
        raise NotImplementedError("RRAnalysis is a static class")

    @staticmethod
    def get_example_index(series):
        """ Example index method, returns the length
        :param series: DataSeries object to filter
        :return: DataSeries object filtered
        """
        assert type(series) is DataSeries
        return len(series.get_series)

    @staticmethod
    def poin_indexes(series):
        """ Returns Poincare' indexes """
        rr = series
        # calculates Poincare' indexes
        xdata, ydata = rr[:-1], rr[1:]
        sd1 = np.std((xdata - ydata) / np.sqrt(2.0), ddof=1)
        sd2 = np.std((xdata + ydata) / np.sqrt(2.0), ddof=1)
        sd12 = sd1 / sd2
        sell = sd1 * sd2 * np.pi
        labels = ['sd1', 'sd2', 'sd12', 'sell']

        return [sd1, sd2, sd12, sell], labels


class RRFilters(DataAnalysis):
    """ Static class containing methods for filtering RR intervals data. """

    def __init__(self):
        raise NotImplementedError("RRFilters is a static class")

    @staticmethod
    def example_filter(series):
        """ Example filter method, does nothing
        :param series: DataSeries object to filter
        :return: DataSeries object filtered
        """
        assert type(series) is DataSeries
        return series

        # xTODO: add analysis scripts like in the example