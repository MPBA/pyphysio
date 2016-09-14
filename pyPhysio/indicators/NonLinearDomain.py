# coding=utf-8
from __future__ import division

from ..BaseIndicator import Indicator as _Indicator
from ..filters.Filters import Diff as _Diff
from ..indicators.TimeDomain import Mean as _Mean, StDev as _StDev
from scipy.spatial.distance import cdist as _cd
import numpy as _np
from ..Parameters import Parameter as _Par

__author__ = 'AleB'


# HRV
class PoincareSD1(_Indicator):
    @classmethod
    def algorithm(cls, data, params):
        """
        Calculates Poincare SD 1 and 2
        @return: (SD1, SD2)
        @rtype: (array, array)
        """
        xd, yd = _np.array(list(data[:-1])), _np.array(list(data[1:]))
        sd1 = _np.std((xd - yd) / _np.sqrt(2.0))
        return sd1


class PoincareSD2(_Indicator):
    @classmethod
    def algorithm(cls, data, params):
        """
        Calculates Poincare SD 1 and 2
        @return: (SD1, SD2)
        @rtype: (array, array)
        """
        xd, yd = _np.array(list(data[:-1])), _np.array(list(data[1:]))
        sd2 = _np.std((xd + yd) / _np.sqrt(2.0))
        return sd2


class PoincareSD1SD2(_Indicator):
    @classmethod
    def algorithm(cls, data, params):
        """
        Calculates Poincare SD 1 and 2
        @return: (SD1, SD2)
        @rtype: (array, array)
        """
        sd1 = PoincareSD1()(data)
        sd2 = PoincareSD1()(data)
        return sd1 / sd2


class PoinEll(_Indicator):
    """
    Calculates the Poincaré Ell. index of the data series.
    """

    @classmethod
    def algorithm(cls, data, params):
        sd1 = PoincareSD1()(data)
        sd2 = PoincareSD2()(data)
        return sd1 * sd2 * _np.pi


class PNNx(_Indicator):
    """
    Calculates the relative frequency (0.0-1.0) of pairs of consecutive IBIs in the data series
    where the difference between the two values is greater than the parameter (threshold).
    """

    @classmethod
    def algorithm(cls, data, params):
        return NNx.algorithm(data, params) / float(len(data))

    _params_descriptors = {
        'threshold': _Par(2, float, 'Threshold to select the subsequent differences', 10, lambda x: x > 0),
    }


class NNx(_Indicator):
    """
    Calculates number of pairs of consecutive values in the data where the difference between is greater than the given
    parameter (threshold).
    """

    @classmethod
    def algorithm(cls, data, params):
        th = params['threshold']
        diff = _Diff()(data)
        return sum(1.0 for x in diff*1000 if x > th)

    _params_descriptors = {
        'threshold': _Par(2, float, 'Threshold to select the subsequent differences', 10, lambda x: x > 0),
    }


class Embed(_Indicator):
    @classmethod
    def algorithm(cls, data, params):
        """
        Calculates the the vector of the sequences of length 'subset_size' of the data
        @return: Data array with shape (l - n + 1, n) having l=len(data) and n=subset_size
        @rtype: array
        """
        # TODO (Ale): Inserire algoritmo piu generale con delay tau
        n = params['dimension']
        t = params['delay']
        # ...
        num = len(data) - n + 1
        if num > 0:
            emb = _np.zeros([num, n])
            for i in xrange(num):
                emb[i, :] = data[i:i + n]
            return emb
        else:
            return []

    # TODO: Inserire algoritmo con delay: tau
    @staticmethod
    def BuildTakensVector(data, m, tau):
        # DataInt = range(1001)
        N = len(data)
        jump = tau
        maxjump = (m - 1) * jump
        jumpsvect = range(0, maxjump + 1, jump)
        # print("jumpsvect: "+str(jumpsvect))
        numjumps = len(jumpsvect)
        numelem = N - maxjump
        # print("Building matrix "+str(numelem)+"x"+str(numjumps))
        DataExp = _np.zeros(shape=(numelem, numjumps))
        for i in range(numelem):
            for j in range(numjumps):
                DataExp[i, j] = data[jumpsvect[j] + i]

    _params_descriptors = {
        'dimension': _Par(2, int, 'Embed dimension', 1, lambda x: x > 0),
        'delay': _Par(2, int, 'Embed delay', 1, lambda x: x > 0)
    }


class ApproxEntropy(_Indicator):
    """
    Calculates the approx entropy of the data series.
    """

    @classmethod
    def algorithm(cls, data, params):
        if len(data) < 3:
            return _np.nan
        else:
            r = params['radius']
            uj_m = Embed(dimension=2, delay=1)(data)
            uj_m1 = Embed(dimension=3, delay=1)(data)
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

    _params_descriptors = {
        'radius': _Par(2, float, 'Radius', 0.5, lambda x: x > 0),
    }


class SampleEntropy(_Indicator):
    """
    Calculates the sample entropy of the data series.
    """

    @classmethod
    def algorithm(cls, data, params):
        if len(data) < 4:
            return _np.nan
        else:
            r = params['radius']
            uj_m = Embed(dimension=2, delay=1)(data)
            uj_m1 = Embed(dimension=3, delay=1)(data)

            num_elem_m = uj_m.shape[0]
            num_elem_m1 = uj_m1.shape[0]

            r = r * _StDev()(data)
            d_m = _cd(uj_m, uj_m,
                      'chebyshev')  # TODO (Ale): mettere questo come algoritmo esterno per sfruttare la cache (usato anche da approxEntropy)
            d_m1 = _cd(uj_m1, uj_m1,
                       'chebyshev')  # TODO (Ale): mettere questo come algoritmo esterno per sfruttare la cache (usato anche da approxEntropy)

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

    _params_descriptors = {
        'radius': _Par(2, float, 'Radius', 0.5, lambda x: x > 0),
    }


class DFAShortTerm(_Indicator):
    """
    Calculate the alpha1 (short term) component index of the De-trended Fluctuation Analysis.
    """

    @classmethod
    def algorithm(cls, data, params):
        # calculates De-trended Fluctuation Analysis: alpha1 (short term) component
        x = data
        if len(x) < 16:
            return _np.nan
        else:
            ave = _Mean()(x)
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


class DFALongTerm(_Indicator):
    """
    Calculate the alpha2 (long term) component index of the De-trended Fluctuation Analysis.
    """

    @classmethod
    def algorithm(cls, data, params):
        # calculates De-trended Fluctuation Analysis: alpha2 (long term) component
        x = data
        if len(x) < 64:
            return _np.nan
        else:
            ave = _Mean()(x)
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


# class FractalDimension(_Indicator): #TODO: sistemare tutto ma nascondere per il momento
#
# #Calculates the fractal dimension of the data series.
#
#
# def __init__(self, params=None, **kwargs):
#         super(FractalDimension, self).__init__(params, kwargs)
#
#     @classmethod
#     def algorithm(cls, data, params):
#         assert 'cra' in params, "This feature needs the parameter 'cra'."
#         assert 'crb' in params, "This feature needs the parameter 'crb'."
#         if len(data) < 3:
#             return _np.nan
#         else:
#             uj_m = OrderedSubsets(dimension=2, delay=1)(data)
#             cra = params['cra']
#             crb = params['crb']
#             mutual_distance = _pd(uj_m, 'chebyshev')
#
#             num_elem = len(mutual_distance)
#
#             rr = _mq(mutual_distance, prob=[cra, crb])
#             ra = rr[0]
#             rb = rr[1]
#
#             cmr_a = (sum(1 for i in mutual_distance if i <= ra)) / num_elem
#             cmr_b = (sum(1 for i in mutual_distance if i <= rb)) / num_elem
#
#             return (_np.log(cmr_b) - _np.log(cmr_a)) / (_np.log(rb) - _np.log(ra))
#
# class SVDEntropy(_Indicator): #TODO: sistemare tutto ma nascondere per il momento
#     #Calculates the SVD entropy of the data series.
#
#
#     def __init__(self, params=None, **kwargs):
#         super(SVDEntropy, self).__init__(params, kwargs)
#
#     @classmethod
#     def algorithm(cls, data, params):
#         if len(data) < 2:
#             return _np.nan
#         else:
#             uj_m = Embed(dimension=2, delay=1)(data)
#             w = _np.linalg.svd(uj_m, compute_uv=False)
#             w /= sum(w)
#             return -1 * sum(w * _np.log(w))
#
# class Fisher(_Indicator): #TODO: sistemare tutto ma nascondere per il momento
#     #Calculates the Fisher Information index of the data series.
#
#     def __init__(self, params=None, **kwargs):
#         super(Fisher, self).__init__(params, kwargs)
#
#     @classmethod
#     def algorithm(cls, data, params):
#         if len(data) < 2:
#             return _np.nan
#         else:
#             uj_m = Embed(dimension=2, delay=1)(data)
#             w = _np.linalg.svd(uj_m, compute_uv=False)
#             w /= sum(w)
#             fi = 0
#             for i in xrange(0, len(w) - 1):  # from Test1 to M
#                 fi += ((w[i + 1] - w[i]) ** 2) / (w[i])
#
#             return fi
#
# class CorrelationDim(_Indicator): #TODO: sistemare tutto ma nascondere per il momento
#     #Calculates the correlation dimension of the data series.
#
#     def __init__(self, params=None, **kwargs):
#         super(CorrelationDim, self).__init__(params, kwargs)
#
#     @classmethod
#     def algorithm(cls, data, params):
#         assert 'corr_dim_len' in params, "This feature needs the parameter 'corr_dim_len'."
#         if len(data) < params['corr_dim_len']:
#             return _np.nan
#         else:
#             rr = data  # rr in seconds
#             # Check also the other indicators to work with seconds!
#             uj = Embed.get(rr, dict(subset_size=params['corr_dim_len']))
#             num_elem = uj.shape[0]
#             r_vector = _np.arange(0.3, 0.46, 0.02)  # settings
#             c = _np.zeros(len(r_vector))
#             jj = 0
#             n = _np.zeros(num_elem)
#             dj = _cd(uj, uj)
#             for r in r_vector:
#                 for i in xrange(num_elem):
#                     vector = dj[i]
#                     n[i] = float(sum(1 for i in vector if i <= r)) / num_elem
#                 c[jj] = _np.sum(n) / num_elem
#                 jj += 1
#
#             log_c = _np.log(c)
#             log_r = _np.log(r_vector)
#
#             return (log_c[-1] - log_c[0]) / (log_r[-1] - log_r[0])
#
# class Hurst(_Indicator): #TODO: sistemare tutto ma nascondere per il momento
#     #Calculates the Hurst HRV index of the data series.
#
#     def __init__(self, params=None, **kwargs):
#         super(Hurst, self).__init__(params, kwargs)
#
#     @classmethod
#     def algorithm(cls, data, params):
#         n = len(data)
#         if n < 2:
#             return _np.nan
#         else:
#             t = _np.arange(1.0, n + 1)
#             y = _np.cumsum(data)
#             ave_t = _np.array(y / t)
#
#             s_t = _np.zeros(n)
#             r_t = _np.zeros(n)
#             for i in xrange(n):
#                 s_t[i] = _np.std(data[:i + 1])
#                 x_t = y - t * ave_t[i]
#                 r_t[i] = _np.max(x_t[:i + 1]) - _np.min(x_t[:i + 1])
#
#             r_s = r_t / s_t
#             r_s = _np.log(r_s)
#             n = _np.log(t).reshape(n, 1)
#             h = _np.linalg.lstsq(n[1:], r_s[1:])[0]
#             return h[0]
#
# class PetrosianFracDim(_Indicator): #TODO: sistemare tutto ma nascondere per il momento
#
#     #Calculates the petrosian's fractal dimension of the data series.
#
#
#     def __init__(self, params=None, **kwargs):
#         super(PetrosianFracDim, self).__init__(params, kwargs)
#
#     @classmethod
#     def algorithm(cls, data, params):
#         d = _Diff()(data)
#         n_delta = 0  # number of sign changes in derivative of the signal
#         for i in xrange(1, len(d)):
#             if d[i] * d[i - 1] < 0:
#                 n_delta += 1
#         n = len(data)
#         return _np.float(_np.log10(n) / (_np.log10(n) + _np.log10(n / n + 0.4 * n_delta)))
