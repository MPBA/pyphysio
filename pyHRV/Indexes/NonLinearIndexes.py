# coding=utf-8

import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.stats.mstats import mquantiles

from pyHRV.Cache import RRDiff, BuildTakensVector2, BuildTakensVector3
from pyHRV.Indexes.Indexes import NonLinearIndex
from pyHRV.Indexes.TDIndexes import RRMean
from pyHRV.utility import build_takens_vector
from pyHRV.PyHRVSettings import PyHRVDefaultSettings as Sett


class ApproxEntropy(NonLinearIndex):
    def __init__(self, data=None):
        super(ApproxEntropy, self).__init__(data)
        R = 0.2  # settings
        uj_m = BuildTakensVector2.get(self._data)
        uj_m1 = BuildTakensVector3.get(self._data)

        card_elem_m = uj_m.shape[0]
        card_elem_m1 = uj_m1.shape[0]

        r = R * np.std(self._data)
        d_m = cdist(uj_m, uj_m, 'chebyshev')
        d_m1 = cdist(uj_m1, uj_m1, 'chebyshev')

        cmr_m_apen = np.zeros(card_elem_m)
        for i in range(card_elem_m):
            vector = d_m[i]
            cmr_m_apen[i] = float((vector <= r).sum()) / card_elem_m

        cmr_m1_apen = np.zeros(card_elem_m1)
        for i in range(card_elem_m1):
            vector = d_m1[i]
            cmr_m1_apen[i] = float((vector <= r).sum()) / card_elem_m1

        Phi_m = np.sum(np.log(cmr_m_apen)) / card_elem_m
        Phi_m1 = np.sum(np.log(cmr_m1_apen)) / card_elem_m1

        self._value = Phi_m - Phi_m1


class SampleEntropy(NonLinearIndex):
    def __init__(self, data=None):
        super(SampleEntropy, self).__init__(data)
        R = Sett.NonLinearIndexes.sample_entropy_r
        uj_m = BuildTakensVector2.get(self._data)
        uj_m1 = BuildTakensVector3.get(self._data)

        numelem_m = uj_m.shape[0]
        numelem_m1 = uj_m1.shape[0]

        r = R * np.std(self._data) #cacheable
        d_m = cdist(uj_m, uj_m, 'chebyshev')
        d_m1 = cdist(uj_m1, uj_m1, 'chebyshev')

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


class FractalDimension(NonLinearIndex):
    def __init__(self, data=None):
        super(FractalDimension, self).__init__(data)
        uj_m = BuildTakensVector2.get(self._data)
        Cra = 0.005 #settings
        Crb = 0.75 #settings
        mutual_distance = pdist(uj_m, 'chebyshev')

        num_elem = len(mutual_distance)

        rr = mquantiles(mutual_distance, prob=[Cra, Crb])
        ra = rr[0]
        rb = rr[1]

        Cmra = float(((mutual_distance <= ra).sum())) / num_elem
        Cmrb = float(((mutual_distance <= rb).sum())) / num_elem

        self._value = (np.log(Cmrb) - np.log(Cmra)) / (np.log(rb) - np.log(ra))


class SVDEntropy(NonLinearIndex):
    def __init__(self, data=None):
        super(SVDEntropy, self).__init__(data)
        uj_m = BuildTakensVector2.get(self._data)
        W = np.linalg.svd(uj_m, compute_uv=False)
        W /= sum(W)
        self._value = -1 * sum(W * np.log(W))


class Fisher(NonLinearIndex):
    def __init__(self, data=None):
        super(Fisher, self).__init__(data)
        uj_m = BuildTakensVector2.get(self._data)
        W = np.linalg.svd(uj_m, compute_uv=False)
        W /= sum(W)
        FI = 0
        for i in xrange(0, len(W) - 1):    # from 1 to M
            FI += ((W[i + 1] - W[i]) ** 2) / (W[i])

        self._value = FI


class CorrDimension(NonLinearIndex):
    def __init__(self, data=None):
        super(CorrDimension, self).__init__(data)
        rr = self._data / 1000  # rr in seconds
        uj = build_takens_vector(rr, Sett.NonLinearIndexes.correlation_dimension_len)
        numelem = uj.shape[0]
        r_vect = np.arange(0.3, 0.46, 0.02)  # settings
        C = np.zeros(len(r_vect))
        jj = 0
        N = np.zeros(numelem)
        dj = cdist(uj, uj, 'euclidean')
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
        super(PoinSD1, self).__init__(data)
        xdata, ydata = self._data[:-1], self._data[1:]
        self._value = np.std((xdata - ydata) / np.sqrt(2.0), ddof=1)


class PoinSD2(NonLinearIndex):
    def __init__(self, data=None):
        super(PoinSD2, self).__init__(data)
        xdata, ydata = self._data[:-1], self._data[1:]
        sd2 = np.std((xdata + ydata) / np.sqrt(2.0), ddof=1)


class PoinSD12(NonLinearIndex):
    def __init__(self, data=None):
        super(PoinSD12, self).__init__(data)
        xdata, ydata = self._data[:-1], self._data[1:]
        sd1 = np.std((xdata - ydata) / np.sqrt(2.0), ddof=1) #cacheable
        sd2 = np.std((xdata + ydata) / np.sqrt(2.0), ddof=1) #cacheable
        self._value = sd1 / sd2


class PoinEll(NonLinearIndex):
    def __init__(self, data=None):
        super(PoinEll, self).__init__(data)
        xdata, ydata = self._data[:-1], self._data[1:]
        sd1 = np.std((xdata - ydata) / np.sqrt(2.0), ddof=1) #cacheable
        sd2 = np.std((xdata + ydata) / np.sqrt(2.0), ddof=1) #cacheable
        self._value = sd1 * sd2 * np.pi


class Hurst(NonLinearIndex):
    def __init__(self, data=None):
        super(Hurst, self).__init__(data)
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
        super(Pfd, self).__init__(data)
        D = RRDiff.get(self._data)
        N_delta = 0  # number of sign changes in derivative of the signal
        for i in xrange(1, len(D)):
            if D[i] * D[i - 1] < 0:
                N_delta += 1
        n = len(self._data)
        self._value = np.float(np.log10(n) / (np.log10(n) + np.log10(n / n + 0.4 * N_delta)))


class DFAShortTerm(NonLinearIndex):
    def __init__(self, data=None):
        super(DFAShortTerm, self).__init__(data)
        #calculates Detrended Fluctuation Analysis: alpha1 (short term) component
        X = self._data
        Ave = RRMean.get(X)
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


class DFALongTerm(NonLinearIndex):
    def __init__(self, data=None):
        super(DFALongTerm, self).__init__(data)
        #calculates Detrended Fluctuation Analysis: alpha2 (long term) component
        X = self._data
        Ave = RRMean.get(X)
        Y = np.cumsum(X)
        Y -= Ave
        lMax = np.min([64, len(X)])
        L = np.arange(4, lMax + 1, 4) ##TODO: check if start from 4 or 16 (Andrea)
        F = np.zeros(len(L))  # F(n) of different given box length n
        for i in xrange(0, len(L)):
            n = int(L[i])  # for each box length L[i]
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

