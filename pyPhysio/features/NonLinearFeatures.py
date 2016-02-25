# coding=utf-8
from __future__ import division

__author__ = 'AleB'

from scipy.spatial.distance import cdist as _cd, pdist as _pd
from scipy.stats.mstats import mquantiles as _mq
import numpy as _np

from CacheOnlyFeatures import OrderedSubsets, PoincareSD
from ..filters.Filters import Diff as _Diff
from ..BaseFeature import Feature as _Feature
from TDFeatures import Mean as _Mean, SD as _SD


class NonLinearFeature(_Feature):
    """
    This is the base class for the Non Linear Features.
    """

    def __init__(self, params=None, _kwargs=None):
        super(NonLinearFeature, self).__init__(params, _kwargs)

    @classmethod
    def algorithm(cls, data, params):
        """
        Placeholder for the subclasses
        @raise NotImplementedError: Ever
        """
        raise NotImplementedError(cls.__name__ + " is a NonLinearFeature but it is not implemented.")


class ApproxEntropy(NonLinearFeature):
    """
    Calculates the approx entropy of the data series.
    """

    def __init__(self, params=None, **kwargs):
        super(ApproxEntropy, self).__init__(params, kwargs)

    @classmethod
    def algorithm(cls, data, params):
        assert 'approx_entropy_r' in params, "This feature needs the parameter 'approx_entropy_r'."
        if len(data) < 3:
            return _np.nan
        else:
            r = params['approx_entropy_r']
            uj_m = OrderedSubsets.get(data, subset_size=2)
            uj_m1 = OrderedSubsets.get(data, subset_size=3)
            card_elem_m = uj_m.shape[0]
            card_elem_m1 = uj_m1.shape[0]

            r = r * _np.std(data)
            d_m = _cd(uj_m, uj_m, 'chebyshev')
            d_m1 = _cd(uj_m1, uj_m1, 'chebyshev')

            cmr_m_ap_en = _np.zeros(card_elem_m)
            for i in xrange(card_elem_m):
                vector = d_m[i]
                cmr_m_ap_en[i] = float(sum(1 for i in vector if i <= r)) / card_elem_m

            cmr_m1_ap_en = _np.zeros(card_elem_m1)
            for i in xrange(card_elem_m1):
                vector = d_m1[i]
                cmr_m1_ap_en[i] = float(sum(1 for i in vector if i <= r)) / card_elem_m1

            phi_m = _np.sum(_np.log(cmr_m_ap_en)) / card_elem_m
            phi_m1 = _np.sum(_np.log(cmr_m1_ap_en)) / card_elem_m1

            return phi_m - phi_m1


class SampleEntropy(NonLinearFeature):
    """
    Calculates the sample entropy of the data series.
    """

    def __init__(self, params=None, **kwargs):
        super(SampleEntropy, self).__init__(params, kwargs)

    @classmethod
    def algorithm(cls, data, params):
        assert 'sample_entropy_r' in params, "This feature needs the parameter 'sample_entropy_r'."
        if len(data) < 4:
            return _np.nan
        else:
            r = params['sample_entropy_r']
            uj_m = OrderedSubsets.get(data, subset_size=2)
            uj_m1 = OrderedSubsets.get(data, subset_size=3)

            num_elem_m = uj_m.shape[0]
            num_elem_m1 = uj_m1.shape[0]

            r = r * _SD.get(data)
            d_m = _cd(uj_m, uj_m, 'che'+'bys'+'hev')
            d_m1 = _cd(uj_m1, uj_m1, 'che'+'bys'+'hev')

            cmr_m_sa_mp_en = _np.zeros(num_elem_m)
            for i in xrange(num_elem_m):
                vector = d_m[i]
                cmr_m_sa_mp_en[i] = (sum(1 for i in vector if i <= r) - 1) / (num_elem_m - 1)

            cmr_m1_sa_mp_en = _np.zeros(num_elem_m1)
            for i in xrange(num_elem_m1):
                vector = d_m1[i]
                cmr_m1_sa_mp_en[i] = (sum(1 for i in vector if i <= r) - 1) / (num_elem_m1 - 1)

            cm = _np.sum(cmr_m_sa_mp_en) / num_elem_m
            cm1 = _np.sum(cmr_m1_sa_mp_en) / num_elem_m1

            return _np.log(cm / cm1)


class FractalDimension(NonLinearFeature):
    """
    Calculates the fractal dimension of the data series.
    """

    def __init__(self, params=None, **kwargs):
        super(FractalDimension, self).__init__(params, kwargs)

    @classmethod
    def algorithm(cls, data, params):
        assert 'cra' in params, "This feature needs the parameter 'cra'."
        assert 'crb' in params, "This feature needs the parameter 'crb'."
        if len(data) < 3:
            return _np.nan
        else:
            uj_m = OrderedSubsets.get(data, subset_size=2)
            cra = params['cra']
            crb = params['crb']
            mutual_distance = _pd(uj_m, 'che'+'bys'+'hev')

            num_elem = len(mutual_distance)

            rr = _mq(mutual_distance, prob=[cra, crb])
            ra = rr[0]
            rb = rr[1]

            cmr_a = (sum(1 for i in mutual_distance if i <= ra)) / num_elem
            cmr_b = (sum(1 for i in mutual_distance if i <= rb)) / num_elem

            return (_np.log(cmr_b) - _np.log(cmr_a)) / (_np.log(rb) - _np.log(ra))


class SVDEntropy(NonLinearFeature):
    """
    Calculates the SVD entropy of the data series.
    """

    def __init__(self, params=None, **kwargs):
        super(SVDEntropy, self).__init__(params, kwargs)

    @classmethod
    def algorithm(cls, data, params):
        if len(data) < 2:
            return _np.nan
        else:
            uj_m = OrderedSubsets.get(data, subset_size=2)
            w = _np.linalg.svd(uj_m, compute_uv=False)
            w /= sum(w)
            return -1 * sum(w * _np.log(w))


class Fisher(NonLinearFeature):
    """
    Calculates the Fisher index of the data series.
    """

    def __init__(self, params=None, **kwargs):
        super(Fisher, self).__init__(params, kwargs)

    @classmethod
    def algorithm(cls, data, params):
        if len(data) < 2:
            return _np.nan
        else:
            uj_m = OrderedSubsets.get(data, subset_size=2)
            w = _np.linalg.svd(uj_m, compute_uv=False)
            w /= sum(w)
            fi = 0
            for i in xrange(0, len(w) - 1):  # from Test1 to M
                fi += ((w[i + 1] - w[i]) ** 2) / (w[i])

            return fi


class CorrelationDim(NonLinearFeature):
    """
    Calculates the correlation dimension of the data series.
    """

    def __init__(self, params=None, **kwargs):
        super(CorrelationDim, self).__init__(params, kwargs)

    @classmethod
    def algorithm(cls, data, params):
        assert 'corr_dim_len' in params, "This feature needs the parameter 'corr_dim_len'."
        if len(data) < params['corr_dim_len']:
            return _np.nan
        else:
            rr = data / 1000  # rr in seconds TODO: wut? semplificabile??
            # Check also the other features to work with seconds!
            uj = OrderedSubsets.get(rr, dict(subset_size=params['corr_dim_len']))
            num_elem = uj.shape[0]
            r_vector = _np.arange(0.3, 0.46, 0.02)  # settings
            c = _np.zeros(len(r_vector))
            jj = 0
            n = _np.zeros(num_elem)
            dj = _cd(uj, uj)
            for r in r_vector:
                for i in xrange(num_elem):
                    vector = dj[i]
                    n[i] = float(sum(1 for i in vector if i <= r)) / num_elem
                c[jj] = _np.sum(n) / num_elem
                jj += 1

            log_c = _np.log(c)
            log_r = _np.log(r_vector)

            return (log_c[-1] - log_c[0]) / (log_r[-1] - log_r[0])


class PoinSD1(NonLinearFeature):
    """
    Calculates the SD1 Poincaré index of the data series.
    """

    def __init__(self, params=None, **kwargs):
        super(PoinSD1, self).__init__(params, kwargs)

    @classmethod
    def algorithm(cls, data, params):
        sd1, sd2 = PoincareSD.get(data)
        return sd1


class PoinSD2(NonLinearFeature):
    """
    Calculates the SD2 Poincaré index of the data series.
    """

    def __init__(self, params=None, **kwargs):
        super(PoinSD2, self).__init__(params, kwargs)

    @classmethod
    def algorithm(cls, data, params):
        sd1, sd2 = PoincareSD.get(data)
        return sd2


class PoinSD12(NonLinearFeature):
    """
    Calculates the ratio between SD1 and SD2 Poincaré features of the data series.
    """

    def __init__(self, params=None, **kwargs):
        super(PoinSD12, self).__init__(params, kwargs)

    @classmethod
    def algorithm(cls, data, params):
        sd1, sd2 = PoincareSD.get(data)
        return sd1 / sd2


class PoinEll(NonLinearFeature):
    """
    Calculates the Poincaré Ell. index of the data series.
    """

    def __init__(self, params=None, **kwargs):
        super(PoinEll, self).__init__(params, kwargs)

    @classmethod
    def algorithm(cls, data, params):
        sd1, sd2 = PoincareSD.get(data)
        return sd1 * sd2 * _np.pi


class Hurst(NonLinearFeature):
    """
    Calculates the Hurst HRV index of the data series.
    """

    def __init__(self, params=None, **kwargs):
        super(Hurst, self).__init__(params, kwargs)

    @classmethod
    def algorithm(cls, data, params):
        n = len(data)
        if n < 2:
            return _np.nan
        else:
            t = _np.arange(1.0, n + 1)
            y = _np.cumsum(data)
            ave_t = _np.array(y / t)

            s_t = _np.zeros(n)
            r_t = _np.zeros(n)
            for i in xrange(n):
                s_t[i] = _np.std(data[:i + 1])
                x_t = y - t * ave_t[i]
                r_t[i] = _np.max(x_t[:i + 1]) - _np.min(x_t[:i + 1])

            r_s = r_t / s_t
            r_s = _np.log(r_s)
            n = _np.log(t).reshape(n, 1)
            h = _np.linalg.lstsq(n[1:], r_s[1:])[0]
            return h[0]


class PetrosianFracDim(NonLinearFeature):
    """
    Calculates the petrosian's fractal dimension of the data series.
    """

    def __init__(self, params=None, **kwargs):
        super(PetrosianFracDim, self).__init__(params, kwargs)

    @classmethod
    def algorithm(cls, data, params):
        d = _Diff.get(data)
        n_delta = 0  # number of sign changes in derivative of the signal
        for i in xrange(1, len(d)):
            if d[i] * d[i - 1] < 0:
                n_delta += 1
        n = len(data)
        return _np.float(_np.log10(n) / (_np.log10(n) + _np.log10(n / n + 0.4 * n_delta)))


class DFAShortTerm(NonLinearFeature):
    """
    Calculate the alpha1 (short term) component index of the De-trended Fluctuation Analysis.
    """

    def __init__(self, params=None, **kwargs):
        super(DFAShortTerm, self).__init__(params, kwargs)

    @classmethod
    def algorithm(cls, data, params):
        # calculates De-trended Fluctuation Analysis: alpha1 (short term) component
        x = data
        if len(x) < 16:
            return _np.nan
        else:
            ave = _Mean.get(x)
            y = _np.cumsum(x)
            y -= ave
            l = _np.arange(4, 17, 4)
            f = _np.zeros(len(l))  # f(n) of different given box length n
            for i in xrange(0, len(l)):
                n = int(l[i])  # for each box length l[i]
                for j in xrange(0, len(x), n):  # for each box
                    if j + n < len(x):
                        c = range(j, j + n)
                        c = _np.vstack([c, _np.ones(n)]).T  # coordinates of time in the box
                        z = y[j:j + n]  # the value of example_data in the box
                        f[i] += _np.linalg.lstsq(c, z)[1]  # add residue in this box
                f[i] /= ((len(x) / n) * n)
            f = _np.sqrt(f)
            return _np.linalg.lstsq(_np.vstack([_np.log(l), _np.ones(len(l))]).T, _np.log(f))[0][0]


class DFALongTerm(NonLinearFeature):
    """
    Calculate the alpha2 (long term) component index of the De-trended Fluctuation Analysis.
    """

    def __init__(self, params=None, **kwargs):
        super(DFALongTerm, self).__init__(params, kwargs)

    @classmethod
    def algorithm(cls, data, params):
        # calculates De-trended Fluctuation Analysis: alpha2 (long term) component
        x = data
        if len(x) < 16:
            return _np.nan
        else:
            ave = _Mean.get(x)
            y = _np.cumsum(x)
            y -= ave
            l_max = _np.min([64, len(x)])
            l = _np.arange(16, l_max + 1, 4)
            f = _np.zeros(len(l))  # f(n) of different given box length n
            for i in xrange(0, len(l)):
                n = int(l[i])  # for each box length l[i]
                for j in xrange(0, len(x), n):  # for each box
                    if j + n < len(x):
                        c = range(j, j + n)
                        c = _np.vstack([c, _np.ones(n)]).T  # coordinates of time in the box
                        z = y[j:j + n]  # the value of example_data in the box
                        f[i] += _np.linalg.lstsq(c, z)[1]  # add residue in this box
                f[i] /= ((len(x) / n) * n)
            f = _np.sqrt(f)

            return _np.linalg.lstsq(_np.vstack([_np.log(l), _np.ones(len(l))]).T, _np.log(f))[0][0]
