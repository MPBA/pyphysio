# coding=utf-8
__author__ = 'AleB'

from PyHRVSettings import PyHRVDefaultSettings as Sett
from DataSeries import *
from utility import *
import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.stats.mstats import mquantiles


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


class NonLinearIndex(Index):
    def __init__(self, interp_freq, data=None, value=None):
        super(NonLinearIndex, self).__init__(data, value)


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
        return np.sum(self._spec_band) / len(self._freq_band)


class PowerInBandNormal(InBand):
    def __init__(self, fmin, fmax, data=None, interp_freq=Sett.interpolation_freq_default):
        super(PowerInBandNormal, self).__init__(fmin, fmax, interp_freq, data=data)
        self._value = (np.sum(self._spec_band) / len(self._freq_band)) / self._total_band


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
        return np.mean(60 / data)


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
        return np.std(60 / data)


## self._value= NNx/len(diff) >> not convenient for a parameter problem
class PNNx(TDIndex):
    def __init__(self, threshold, data=None):
        super(TDIndex, self).__init__(data)
        self._xth = threshold
        self._value = NNx(threshold, data).value / len(data)


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
        self._value = np.sqrt(sum(diff ** 2) / (len(diff) - 1))


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
        self._value = LF(self._data).value / (HF(self._data).value + LF(self._data).value)


class NormalHF(FDIndex):
    def __init__(self, data=None):
        super(FDIndex, self).__init__(data)
        self._value = HF(self._data).value / (HF(self._data).value + LF(self._data).value)


#############
# NLIN DOMAIN
#############

class ApEn(NonLinearIndex):
    def __init__(self, data=None):
        super(ApEn, self).__init__(data)
        R = 0.2  # settings
        Uj_m = build_takens_vector(self._data, 2)  # cacheable
        Uj_m1 = build_takens_vector(self._data, 3)  # cacheable

        numelem_m = Uj_m.shape[0]
        numelem_m1 = Uj_m1.shape[0]

        r = R * np.std(self._data)  # cacheable
        d_m = cdist(Uj_m, Uj_m, 'chebyshev')
        d_m1 = cdist(Uj_m1, Uj_m1, 'chebyshev')

        Cmr_m_ApEn = np.zeros(numelem_m)
        for i in range(numelem_m):
            vector = d_m[i]
            Cmr_m_ApEn[i] = float((vector <= r).sum()) / numelem_m

        Cmr_m1_ApEn = np.zeros(numelem_m1)
        for i in range(numelem_m1):
            vector = d_m1[i]
            Cmr_m1_ApEn[i] = float((vector <= r).sum()) / numelem_m1

        Phi_m = np.sum(np.log(Cmr_m_ApEn)) / numelem_m
        Phi_m1 = np.sum(np.log(Cmr_m1_ApEn)) / numelem_m1

        self._value = Phi_m - Phi_m1


class SampEn(NonLinearIndex):
    def __init__(self, data=None):
        R = 0.2 #settings
        Uj_m = build_takens_vector(self._data, 2) #cacheable
        Uj_m1 = build_takens_vector(self._data, 3) #cacheable

        numelem_m = Uj_m.shape[0]
        numelem_m1 = Uj_m1.shape[0]

        r = R * np.std(self._data) #cacheable
        d_m = cdist(Uj_m, Uj_m, 'chebyshev')
        d_m1 = cdist(Uj_m1, Uj_m1, 'chebyshev')

        Cmr_m_SampEn = np.zeros(numelem_m)
        for i in range(numelem_m):
            vector = d_m[i]
            Cmr_m_SampEn[i] = float((vector <= r).sum() - 1) / (numelem_m - 1)

        Cmr_m1_SampEn = np.zeros(numelem_m1)
        for i in range(numelem_m1):
            vector = d_m1[i]
            Cmr_m1_SampEn[i] = float((vector <= r).sum() - 1) / (numelem_m1 - 1)

        Cm = np.sum(Cmr_m_SampEn) / numelem_m
        Cm1 = np.sum(Cmr_m1_SampEn) / numelem_m1

        self._value = np.log(Cm / Cm1)


class FracDim(NonLinearIndex):
    def __init__(self, data=None):
        Uj_m = build_takens_vector(self._data, 2) #cacheable
        Cra = 0.005 #settings
        Crb = 0.75 #settings
        mutualDistance = pdist(Uj_m, 'chebyshev')

        numelem = len(mutualDistance)

        rr = mquantiles(mutualDistance, prob=[Cra, Crb])
        ra = rr[0]
        rb = rr[1]

        Cmra = float(((mutualDistance <= ra).sum())) / numelem
        Cmrb = float(((mutualDistance <= rb).sum())) / numelem

        self._value = (np.log(Cmrb) - np.log(Cmra)) / (np.log(rb) - np.log(ra))


class SVDEn(NonLinearIndex):
    def __init__(self, data=None):
        Uj_m = build_takens_vector(self._data, 2) #cacheable
        W = np.linalg.svd(Uj_m, compute_uv=0)
        W /= sum(W)
        self._value = -1 * sum(W * np.log(W))


class Fisher(NonLinearIndex):
    def __init__(self, data=None):
        Uj_m = build_takens_vector(self._data, 2) #cacheable
        W = np.linalg.svd(Uj_m, compute_uv=0)
        W /= sum(W)
        FI = 0
        for i in xrange(0, len(W) - 1):    # from 1 to M
            FI += ((W[i + 1] - W[i]) ** 2) / (W[i])

        self._value = FI


class CorrDim(NonLinearIndex):
    def __init__(self, data=None):
        LEN = 10 #settings
        rr = self._data / 1000 # rr in seconds
        Uj = build_takens_vector(rr, LEN)
        numelem = Uj.shape[0]
        r_vect = np.arange(0.3, 0.46, 0.02) #settings
        C = np.zeros(len(r_vect))
        jj = 0
        N = np.zeros(numelem)
        dj = cdist(Uj, Uj, 'euclidean')
        for r in r_vect:
            for i in range(numelem):
                vector = dj[i]
                N[i] = float((vector <= r).sum()) / numelem
            C[jj] = np.sum(N) / numelem
            jj += 1

        logC = np.log(C)
        logr = np.log(r_vect)

        self._value = (logC[-1] - logC[0]) / (logr[-1] - logr[0])


class PoinSD1(NonLinearIndex):
    def __init__(self, data=None):
        xdata, ydata = self._data[:-1], self._data[1:]
        self._value = np.std((xdata - ydata) / np.sqrt(2.0), ddof=1)


class PoinSD2(NonLinearIndex):
    def __init__(self, data=None):
        xdata, ydata = self._data[:-1], self._data[1:]
        sd2 = np.std((xdata + ydata) / np.sqrt(2.0), ddof=1)


class PoinSD12(NonLinearIndex):
    def __init__(self, data=None):
        xdata, ydata = self._data[:-1], self._data[1:]
        sd1 = np.std((xdata - ydata) / np.sqrt(2.0), ddof=1) #cacheable
        sd2 = np.std((xdata + ydata) / np.sqrt(2.0), ddof=1) #cacheable
        self._value = sd1 / sd2


class PoinEll(NonLinearIndex):
    def __init__(self, data=None):
        xdata, ydata = self._data[:-1], self._data[1:]
        sd1 = np.std((xdata - ydata) / np.sqrt(2.0), ddof=1) #cacheable
        sd2 = np.std((xdata + ydata) / np.sqrt(2.0), ddof=1) #cacheable
        self._value = sd1 * sd2 * np.pi


class Hurst(NonLinearIndex):
    def __init__(self, data=None):
        X = self._data
        #calculates hurst exponent
        N = len(X)
        T = np.array([float(i) for i in xrange(1, N + 1)])
        Y = np.cumsum(X)
        Ave_T = Y / T

        S_T = np.zeros((N))
        R_T = np.zeros((N))
        for i in xrange(N):
            S_T[i] = np.std(X[:i + 1])
            X_T = Y - T * Ave_T[i]
            R_T[i] = np.max(X_T[:i + 1]) - np.min(X_T[:i + 1])

        R_S = R_T / S_T
        R_S = np.log(R_S)
        n = np.log(T).reshape(N, 1)
        H = np.linalg.lstsq(n[1:], R_S[1:])[0]
        self._value = H[0]


class Pfd(NonLinearIndex):
    #calculates petrosian fractal dimension
    def __init__(self, data=None):
        D = RRDiff.get(self._data)
        N_delta = 0; #number of sign changes in derivative of the signal
        for i in xrange(1, len(D)):
            if D[i] * D[i - 1] < 0:
                N_delta += 1
        n = len(self._data)
        self._value = np.float(np.log10(n) / (np.log10(n) + np.log10(n / n + 0.4 * N_delta)))


class Dfa_a1(NonLinearIndex):
    def __init__(self, data=None):
        #calculates Detrended Fluctuation Analysis: alpha1 (short term) component
        X = self._data
        Ave = np.mean(X) #cacheable
        Y = np.cumsum(X)
        Y -= Ave

        lunghezza = len(X)
        L = np.arange(4, 17, 4)
        F = np.zeros(len(L)) # F(n) of different given box length n
        for i in xrange(0, len(L)):
            n = int(L[i]) # for each box length L[i]
            for j in xrange(0, len(X), n): # for each box
                if j + n < len(X):
                    c = range(j, j + n)
                    c = np.vstack([c, np.ones(n)]).T      # coordinates of time in the box
                    y = Y[j:j + n]                    # the value of data in the box
                    F[i] += np.linalg.lstsq(c, y)[1]    # add residue in this box
            F[i] /= ((len(X) / n) * n)
        F = np.sqrt(F)
        try:
            Alpha1 = np.linalg.lstsq(np.vstack([np.log(L), np.ones(len(L))]).T, np.log(F))[0][0]
        except ValueError:
            Alpha1 = np.nan
        self._value = Alpha1


class Dfa_a2(NonLinearIndex):
    def __init__(self, data=None):
        #calculates Detrended Fluctuation Analysis: alpha2 (long term) component
        X = self._data
        Ave = np.mean(X) #cacheable
        Y = np.cumsum(X)
        Y -= Ave
        lMax = np.min([64, len(X)])
        L = np.arange(4, lMax + 1, 4) ##TODO: check if start from 4 or 16 (Andrea)
        F = np.zeros(len(L)) # F(n) of different given box length n
        for i in xrange(0, len(L)):
            n = int(L[i]) # for each box length L[i]
            for j in xrange(0, len(X), n): # for each box
                if j + n < len(X):
                    c = range(j, j + n)
                    c = np.vstack([c, np.ones(n)]).T      # coordinates of time in the box
                    y = Y[j:j + n]                    # the value of data in the box
                    F[i] += np.linalg.lstsq(c, y)[1]    # add residue in this box
            F[i] /= ((len(X) / n) * n)
        F = np.sqrt(F)
        try:
            Alpha2 = np.linalg.lstsq(np.vstack([np.log(L), np.ones(len(L))]).T, np.log(F))[0][0]
        except ValueError:
            Alpha2 = np.nan

        self._value = Alpha2


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