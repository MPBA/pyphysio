# coding=utf-8
__author__ = 'AleB'

from DataSeries import *
import numpy as np
from utility import power, interpolate_rr
from scipy import signal
from PyHRVSettings import PyHRVDefaultSettings as Sett

# TODO: comments
# 1. Sarà necessario rendere le classi degli indici da mettere in cache figlie anche di CacheableDataCalc
#    in modo da poter chiamare ad es. in HRMean.calculate un self._value = 60/RRMean.get(..)
# 2. Spostare i calcoli nel costruttore es. HRMean !NB: data non dovrebbe essere None
# 3. Per far funzionare tutti i TD deve andare il punto 1.
# 4. Implementare gli indici

# sostituito nella classe Index il metodo value con calculate

class DataAnalysis(object):
    pass


class Index(object):
    def __init__(self, data=None):
        self._value = None
        self._data = data

    @property
    def calculated(self):
        """
        Returns weather the index is already calculated and up-to-date
        @return: Boolean
        """
        return not (self._value is None)

    @property
    def calculate(self):
        return self._value

    def update(self, data):
        self._data = data
        self._value = None


class TDIndex(Index):
    def __init__(self, data=None):
        super(TDIndex, self).__init__(data)


class FDIndex(Index):
    def __init__(self, data=None):
        super(FDIndex, self).__init__(data)

    def _interpolate(self, to_freq):
        """
        Privata. Interpola quando chiamata dalle sottoclassi
        @param to_freq:
        @return:
        """
        rr_interp, bt_interp = interpolate_rr(self._data, to_freq)
        self._data = DataSeries(rr_interp)  # TODO: perdita della cache

    def _estimate_psd(self, fsamp):
        # TODO: estimate PSD with vary methods
        ### welch
        freqs, spect = signal.welch(self._data, fsamp)
        spect = np.sqrt(spect)
        return freqs, spect/np.max(spect)


class PowerInBand(FDIndex):
    def __init__(self, fmin, fmax, data=None):
        super(PowerInBand, self).__init__(data)
        self._fmin = fmin
        self._fmax = fmax

        freq, spec, Total=###

        indexes = np.array([i for i in range(len(spec)) if freq[i] >= fmin and freq[i]<fmax])
        freq_band = freq[indexes]
        self._value = np.sum(freq_band)/len(freq_band)


class PowerInBandNormal(FDIndex):
    def __init__(self, fmin, fmax, data=None):
        super(PowerInBand, self).__init__(data)
        self._fmin = fmin
        self._fmax = fmax

        freq, spec, Total=###

        indexes = np.array([i for i in range(len(spec)) if freq[i] >= fmin and freq[i]<fmax])
        freq_band = freq[indexes]
        self._value = (np.sum(freq_band)/len(freq_band))/Total


class PeakInBand(FDIndex):
    def __init__(self, fmin, fmax, data=None):
        super(PeakInBand, self).__init__(data)
        self._fmin = fmin
        self._fmax = fmax

        freq, spec, Total=###

        indexes = np.array([i for i in range(len(spec)) if freq[i] >= fmin and freq[i]<fmax])
        freq_band = freq[indexes]
        spec_band = spec[indexes]
        self._value = freq_band[np.argmax(freq_band)]


#############
# TIME DOMAIN
#############
class RRMean(TDIndex):
    def __init__(self, data=None):
        super(TDIndex, self).__init__(data)
        self._value = np.mean(self._data)


class HRMean(TDIndex):
    def __init__(self, data=None):
        super(TDIndex, self).__init__(data)

class RRmedian(TDIndex):
    def __init__(self, data=None):
        super(TDIndex, self).__init__(data)
        self._value = np.median(self._data)

class RRSTD(TDIndex):
    def __init__(self, data=None):
        super(TDIndex, self).__init__(data)
        self._value = np.std(self._data)


class HRSTD(TDIndex):
    def __init__(self, data=None):
        super(TDIndex, self).__init__(data)


class pNNx(TDIndex):
    def __init__(self, xthreshold, data=None):
        super(TDIndex, self).__init__(data)
        self._xth = xthreshold
        diff = RRDiff.get(self._data)
        self._value = 100.0 * sum(1 for x in diff if x > self._xth) / len(diff)


class NNx(TDIndex):
    def __init__(self, xthreshold, data=None):
        super(TDIndex, self).__init__(data)
        self._xth = xthreshold
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


class LF(PowerInBand):
    def __init__(self, data=None):
        super(LF, self).__init__(Sett.bands_vlf_upper_bound, Sett.bands_lf_upper_bound, data)


class HF(PowerInBand):
    def __init__(self, data=None):
        super(HF, self).__init__(Sett.bands_lf_upper_bound, Sett.bands_hf_upper_bound, data)


class Total(PowerInBand):
    def __init__(self, data=None):
        super(Total, self).__init__(Sett.bands_vlf_lower_bound, Sett.bands_lf_upper_bound, data)


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
        #LF/HF


class NormalLF(FDIndex):
    def __init__(self, data=None):
        super(FDIndex, self).__init__(data)
        #LF/(LF+HF)


class NormalHF(FDIndex):
    def __init__(self, data=None):
        super(FDIndex, self).__init__(data)
        # HF/(LF+HF)


#############
# NLIN DOMAIN
#############
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
        RR = series
        # calculates Poincare' indexes
        xdata, ydata = RR[:-1], RR[1:]
        sd1 = np.std((xdata - ydata) / np.sqrt(2.0), ddof=1)
        sd2 = np.std((xdata + ydata) / np.sqrt(2.0), ddof=1)
        sd12 = sd1 / sd2
        sEll = sd1 * sd2 * np.pi
        labels = ['sd1', 'sd2', 'sd12', 'sEll']

        return [sd1, sd2, sd12, sEll], labels


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

        # TODO: add analysis scripts like in the example