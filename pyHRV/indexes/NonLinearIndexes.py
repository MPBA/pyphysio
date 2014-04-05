__all__ = ['ApproxEntropy', 'CorrelationDim', 'DFALongTerm', 'DFAShortTerm', 'Fisher', 'FractalDimension', 'Hurst',
           'PetrosianFracDim', 'PoinEll', 'PoinSD1', 'PoinSD12', 'PoinSD2', 'SVDEntropy', 'SampleEntropy']

import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.stats.mstats import mquantiles

from pyHRV.Cache import RRDiff, BuildTakensVector2, BuildTakensVector3, PoinSD
from pyHRV.indexes.BaseIndexes import NonLinearIndex
from pyHRV.indexes.TDIndexes import Mean
from pyHRV.utility import build_takens_vector
from pyHRV.PyHRVSettings import PyHRVDefaultSettings as Sett


class ApproxEntropy(NonLinearIndex):
    def __init__(self, data=None):
        super(ApproxEntropy, self).__init__(data)
        r = Sett.NonLinearIndexes.approx_entropy_r
        uj_m = BuildTakensVector2.get(self._data)
        uj_m1 = BuildTakensVector3.get(self._data)

        card_elem_m = uj_m.shape[0]
        card_elem_m1 = uj_m1.shape[0]

        r = r * np.std(self._data)
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

        phi_m = np.sum(np.log(cmr_m_apen)) / card_elem_m
        phi_m1 = np.sum(np.log(cmr_m1_apen)) / card_elem_m1

        self._value = phi_m - phi_m1


class SampleEntropy(NonLinearIndex):
    def __init__(self, data=None):
        super(SampleEntropy, self).__init__(data)
        r = Sett.NonLinearIndexes.sample_entropy_r
        uj_m = BuildTakensVector2.get(self._data)
        uj_m1 = BuildTakensVector3.get(self._data)

        num_elem_m = uj_m.shape[0]
        num_elem_m1 = uj_m1.shape[0]

        r = r * np.std(self._data)  #cacheable
        d_m = cdist(uj_m, uj_m, 'chebyshev')
        d_m1 = cdist(uj_m1, uj_m1, 'chebyshev')

        cmr_m_samp_en = np.zeros(num_elem_m)
        for i in range(num_elem_m):
            vector = d_m[i]
            cmr_m_samp_en[i] = float((vector <= r).sum() - 1) / (num_elem_m - 1)

        cmr_m1_samp_en = np.zeros(num_elem_m1)
        for i in range(num_elem_m1):
            vector = d_m1[i]
            cmr_m1_samp_en[i] = float((vector <= r).sum() - 1) / (num_elem_m1 - 1)

        cm = np.sum(cmr_m_samp_en) / num_elem_m
        cm1 = np.sum(cmr_m1_samp_en) / num_elem_m1

        self._value = np.log(cm / cm1)


class FractalDimension(NonLinearIndex):
    def __init__(self, data=None):
        super(FractalDimension, self).__init__(data)
        uj_m = BuildTakensVector2.get(self._data)
        cra = Sett.NonLinearIndexes.fractal_dimension_cra
        crb = Sett.NonLinearIndexes.fractal_dimension_crb
        mutual_distance = pdist(uj_m, 'chebyshev')

        num_elem = len(mutual_distance)

        rr = mquantiles(mutual_distance, prob=[cra, crb])
        ra = rr[0]
        rb = rr[1]

        cmr_a = float(((mutual_distance <= ra).sum())) / num_elem
        cmr_b = float(((mutual_distance <= rb).sum())) / num_elem

        self._value = (np.log(cmr_b) - np.log(cmr_a)) / (np.log(rb) - np.log(ra))


class SVDEntropy(NonLinearIndex):
    def __init__(self, data=None):
        super(SVDEntropy, self).__init__(data)
        uj_m = BuildTakensVector2.get(self._data)
        w = np.linalg.svd(uj_m, compute_uv=False)
        w /= sum(w)
        self._value = -1 * sum(w * np.log(w))


class Fisher(NonLinearIndex):
    def __init__(self, data=None):
        super(Fisher, self).__init__(data)
        uj_m = BuildTakensVector2.get(self._data)
        w = np.linalg.svd(uj_m, compute_uv=False)
        w /= sum(w)
        fi = 0
        for i in xrange(0, len(w) - 1):  # from 1 to M
            fi += ((w[i + 1] - w[i]) ** 2) / (w[i])

        self._value = fi


class CorrelationDim(NonLinearIndex):
    def __init__(self, data=None):
        super(CorrelationDim, self).__init__(data)
        rr = self._data / 1000  # rr in seconds
        uj = build_takens_vector(rr, Sett.NonLinearIndexes.correlation_dimension_len)
        num_elem = uj.shape[0]
        r_vector = np.arange(0.3, 0.46, 0.02)  # settings TODO: arange in settings? (Andrea)
        c = np.zeros(len(r_vector))
        jj = 0
        n = np.zeros(num_elem)
        dj = cdist(uj, uj, 'euclidean')
        for r in r_vector:
            for i in range(num_elem):
                vector = dj[i]
                n[i] = float((vector <= r).sum()) / num_elem
            c[jj] = np.sum(n) / num_elem
            jj += 1

        log_c = np.log(c)
        log_r = np.log(r_vector)

        self._value = (log_c[-1] - log_c[0]) / (log_r[-1] - log_r[0])


class PoinSD1(NonLinearIndex):
    def __init__(self, data=None):
        super(PoinSD1, self).__init__(data)
        sd1, sd2 = PoinSD.get(self._data)
        # TODO: Is this return right? (Andrea)
        self._value = sd1


class PoinSD2(NonLinearIndex):
    def __init__(self, data=None):
        super(PoinSD2, self).__init__(data)
        sd1, sd2 = PoinSD.get(self._data)
        # TODO: Is this return right? (Andrea)
        self._value = sd2


class PoinSD12(NonLinearIndex):
    def __init__(self, data=None):
        super(PoinSD12, self).__init__(data)
        sd1, sd2 = PoinSD.get(self._data)
        self._value = sd1 / sd2


class PoinEll(NonLinearIndex):
    def __init__(self, data=None):
        super(PoinEll, self).__init__(data)
        sd1, sd2 = PoinSD.get(self._data)
        self._value = sd1 * sd2 * np.pi


class Hurst(NonLinearIndex):
    def __init__(self, data=None):
        super(Hurst, self).__init__(data)
        n = len(self._data)
        t = np.array([float(i) for i in xrange(1, n + 1)])
        y = np.cumsum(self._data)
        ave_t = y / t

        s_t = np.zeros(n)
        r_t = np.zeros(n)
        for i in xrange(n):
            s_t[i] = np.std(self._data[:i + 1])
            x_t = y - t * ave_t[i]
            r_t[i] = np.max(x_t[:i + 1]) - np.min(x_t[:i + 1])

        r_s = r_t / s_t
        r_s = np.log(r_s)
        n = np.log(t).reshape(n, 1)
        h = np.linalg.lstsq(n[1:], r_s[1:])[0]
        self._value = h[0]


class PetrosianFracDim(NonLinearIndex):
    # calculates petrosian fractal dimension
    def __init__(self, data=None):
        super(PetrosianFracDim, self).__init__(data)
        d = RRDiff.get(self._data)
        n_delta = 0  # number of sign changes in derivative of the signal
        for i in xrange(1, len(d)):
            if d[i] * d[i - 1] < 0:
                n_delta += 1
        n = len(self._data)
        self._value = np.float(np.log10(n) / (np.log10(n) + np.log10(n / n + 0.4 * n_delta)))


class DFAShortTerm(NonLinearIndex):
    def __init__(self, data=None):
        super(DFAShortTerm, self).__init__(data)
        #calculates De-trended Fluctuation Analysis: alpha1 (short term) component
        x = self._data
        ave = Mean.get(x)
        y = np.cumsum(x)
        y -= ave

        l = np.arange(4, 17, 4)
        f = np.zeros(len(l))  # f(n) of different given box length n
        for i in xrange(0, len(l)):
            n = int(l[i])  # for each box length l[i]
            for j in xrange(0, len(x), n):  # for each box
                if j + n < len(x):
                    c = range(j, j + n)
                    c = np.vstack([c, np.ones(n)]).T  # coordinates of time in the box
                    y = y[j:j + n]  # the value of data in the box
                    f[i] += np.linalg.lstsq(c, y)[1]  # add residue in this box
            f[i] /= ((len(x) / n) * n)
        f = np.sqrt(f)
        try:
            alpha1 = np.linalg.lstsq(np.vstack([np.log(l), np.ones(len(l))]).T, np.log(f))[0][0]
        except ValueError:
            alpha1 = np.nan
        self._value = alpha1


class DFALongTerm(NonLinearIndex):
    def __init__(self, data=None):
        super(DFALongTerm, self).__init__(data)
        #calculates De-trended Fluctuation Analysis: alpha2 (long term) component
        x = self._data
        ave = Mean.get(x)
        y = np.cumsum(x)
        y -= ave
        l_max = np.min([64, len(x)])
        l = np.arange(4, l_max + 1, 4)  # TODO: check if start from 4 or 16 (Andrea)
        f = np.zeros(len(l))  # f(n) of different given box length n
        for i in xrange(0, len(l)):
            n = int(l[i])  # for each box length l[i]
            for j in xrange(0, len(x), n):  # for each box
                if j + n < len(x):
                    c = range(j, j + n)
                    c = np.vstack([c, np.ones(n)]).T  # coordinates of time in the box
                    y = y[j:j + n]  # the value of data in the box
                    f[i] += np.linalg.lstsq(c, y)[1]  # add residue in this box
            f[i] /= ((len(x) / n) * n)
        f = np.sqrt(f)
        try:
            alpha2 = np.linalg.lstsq(np.vstack([np.log(l), np.ones(len(l))]).T, np.log(f))[0][0]
        except ValueError:
            alpha2 = np.nan

        self._value = alpha2
