# coding=utf-8
from __future__ import division

from ..BaseIndicator import Indicator as _Indicator
from ..tools.Tools import Diff as _Diff
from ..indicators.TimeDomain import Mean as _Mean, StDev as _StDev
from scipy.spatial.distance import cdist as _cd
import numpy as _np

__author__ = 'AleB'


class PoincareSD1(_Indicator):
    """
    Return the SD1 value of the Poincare' plot of input Inter Beat Intervals

    Returns
    -------
    sd1 : float
        SD1 of Poincare' plot
    
    """

    def __init__(self, **kwargs):
        _Indicator.__init__(self, **kwargs)

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
    """
    Return the SD2 value of the Poincare' plot of input Inter Beat Intervals

    Returns
    -------
    sd2 : float
        SD2 of Poincare' plot
    
    """

    def __init__(self, **kwargs):
        _Indicator.__init__(self, **kwargs)

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
    """
    Return the SD1/SD2 value of the Poincare' plot of input Inter Beat Intervals

    Returns
    -------
    sd12 : float
        SD1/SD2 of Poincare' plot
    
    """

    def __init__(self, **kwargs):
        _Indicator.__init__(self, **kwargs)

    @classmethod
    def algorithm(cls, data, params):
        """
        Calculates Poincare SD 1 and 2
        @return: (SD1, SD2)
        @rtype: (array, array)
        """
        sd1 = PoincareSD1()(data)
        sd2 = PoincareSD2()(data)
        return sd1 / sd2


class PoinEll(_Indicator):
    """
    Return the SD1*SD2*pi value of the Poincare' plot of input Inter Beat Intervals

    Returns
    -------
    sdEll : float
        SD1*SD2*pi of Poincare' plot
    
    """

    def __init__(self, **kwargs):
        _Indicator.__init__(self, **kwargs)

    @classmethod
    def algorithm(cls, data, params):
        sd1 = PoincareSD1()(data)
        sd2 = PoincareSD2()(data)
        return sd1 * sd2 * _np.pi


class PNNx(_Indicator):
    """
    Computes the relative frequency of pairs of consecutive samples s1, s2 such that s1-s2 >= 'threshold' in
    milliseconds.
    
    Parameters
    ----------
    threshold : int, >0
        Threshold in milliseconds.
        
    Returns
    -------
    PNNx : float
        Relative frequency
    """

    def __init__(self, threshold, **kwargs):
        _Indicator.__init__(self, threshold=threshold, **kwargs)

    @classmethod
    def algorithm(cls, data, params):
        return NNx.algorithm(data, params) / float(len(data))


class NNx(_Indicator):
    """
    Counts the pairs of consecutive samples s1, s2 such that s1-s2 >= 'threshold' in milliseconds.
    
    Parameters
    ----------
    threshold : int, >0
        Threshold in milliseconds.
    """

    def __init__(self, threshold, **kwargs):
        assert threshold > 0, "Not implemented for threshold not > 0"
        _Indicator.__init__(self, threshold=threshold, **kwargs)

    @classmethod
    def algorithm(cls, signal, params):
        th = params['threshold']
        diff = _Diff()(signal)
        return sum(1.0 for x in diff * 1000 if x > th)


class _Embed(_Indicator):
    def __init__(self, dimension, **kwargs):
        _Indicator.__init__(self, dimension=dimension, **kwargs)

    @classmethod
    def algorithm(cls, signal, params):
        """
        Calculates the the vector of the sequences of length 'subset_size' of the data
        @return: Data array with shape (l - n + 1, n) having l=len(data) and n=subset_size
        @rtype: array
        """
        n = params['dimension']
        # t = params['delay']
        num = len(signal) - n + 1
        if num > 0:
            emb = _np.zeros([num, n])
            # 2>3 xrange>range
            for i in range(num):
                emb[i, :] = signal[i:i + n]
            return emb
        else:
            return []


class ApproxEntropy(_Indicator):
    """
    Calculates Approximate Entropy
        
    Optional Parameters
    ----------
    radius : float, >0, default=0.5
        Radius to threshold the distance between the embedded vectors
        
    Returns
    -------
    apen : float
        Approximate Entropy
    """

    def __init__(self, radius=.5, **kwargs):
        assert radius > 0, "Parameter radius should be > 0"
        _Indicator.__init__(self, radius=radius, **kwargs)

    @classmethod
    def algorithm(cls, data, params):
        if len(data) < 3:
            return _np.nan
        else:
            r = params['radius']
            uj_m = _Embed(dimension=2, delay=1)(data)
            uj_m1 = _Embed(dimension=3, delay=1)(data)
            card_elem_m = uj_m.shape[0]
            card_elem_m1 = uj_m1.shape[0]

            r = r * _np.std(data)
            d_m = _cd(uj_m, uj_m, 'chebyshev')
            d_m1 = _cd(uj_m1, uj_m1, 'chebyshev')

            cmr_m_ap_en = _np.zeros(card_elem_m)
            # 2>3 xrange>range
            for i in range(card_elem_m):
                vector = d_m[i]
                cmr_m_ap_en[i] = float(sum(1 for i in vector if i <= r)) / card_elem_m

            cmr_m1_ap_en = _np.zeros(card_elem_m1)
            # 2>3 xrange>range
            for i in range(card_elem_m1):
                vector = d_m1[i]
                cmr_m1_ap_en[i] = float(sum(1 for i in vector if i <= r)) / card_elem_m1

            phi_m = _np.sum(_np.log(cmr_m_ap_en)) / card_elem_m
            phi_m1 = _np.sum(_np.log(cmr_m1_ap_en)) / card_elem_m1

            return phi_m - phi_m1


class SampleEntropy(_Indicator):
    """
    Calculates Sample Entropy
        
    Optional Parameters
    ----------
    radius : float, >0, default=0.5
        Radius to threshold the distance between the embedded vectors
        
    Returns
    -------
    sampen : float
        Sample Entropy
    """

    def __init__(self, radius=.5, **kwargs):
        assert radius > 0, "Parameter radius should be > 0"
        _Indicator.__init__(self, radius=radius, **kwargs)

    @classmethod
    def algorithm(cls, data, params):
        if len(data) < 4:
            return _np.nan
        else:
            r = params['radius']
            uj_m = _Embed(dimension=2, delay=1)(data)
            uj_m1 = _Embed(dimension=3, delay=1)(data)

            num_elem_m = uj_m.shape[0]
            num_elem_m1 = uj_m1.shape[0]

            r = r * _StDev()(data)

            d_m = _cd(uj_m, uj_m, 'chebyshev')
            d_m1 = _cd(uj_m1, uj_m1, 'chebyshev')

            cmr_m_sa_mp_en = _np.zeros(num_elem_m)
            # 2>3 xrange>range
            for i in range(num_elem_m):
                vector = d_m[i]
                cmr_m_sa_mp_en[i] = (sum(1 for i in vector if i <= r) - 1) / (num_elem_m - 1)

            cmr_m1_sa_mp_en = _np.zeros(num_elem_m1)
            # 2>3 xrange>range
            for i in range(num_elem_m1):
                vector = d_m1[i]
                cmr_m1_sa_mp_en[i] = (sum(1 for i in vector if i <= r) - 1) / (num_elem_m1 - 1)

            cm = _np.sum(cmr_m_sa_mp_en) / num_elem_m
            cm1 = _np.sum(cmr_m1_sa_mp_en) / num_elem_m1

            return _np.log(cm / cm1)


class DFAShortTerm(_Indicator):
    """
    Calculate the alpha1 (short term) component index of the De-trended Fluctuation Analysis.
   
    Returns
    -------
    alpha1 : float
        Short term component index of the De-trended Fluctuation Analysis
    """

    def __init__(self, **kwargs):
        _Indicator.__init__(self, **kwargs)

    @classmethod
    def algorithm(cls, data, params):

        x = data
        if len(x) < 16:
            return _np.nan
        else:
            ave = float(_Mean()(x))
            y = _np.cumsum(x).astype(float)
            y -= ave
            l = _np.arange(4, 17, 4)
            f = _np.zeros(len(l))  # f(n) of different given box length n
            # 2>3 xrange>range
            for i in range(0, len(l)):
                n = int(l[i])  # for each box length l[i]
                # 2>3 xrange>range
                for j in range(0, len(x), n):  # for each box
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
   
    Returns
    -------
    alpha2 : float
        Long term component index of the De-trended Fluctuation Analysis
    """

    def __init__(self, **kwargs):
        _Indicator.__init__(self, **kwargs)

    @classmethod
    def algorithm(cls, data, params):
        x = data
        if len(x) < 64:
            return _np.nan
        else:
            ave = float(_Mean()(x))
            y = _np.cumsum(x).astype(float)
            y -= ave
            l_max = _np.min([64, len(x)])
            l = _np.arange(16, l_max + 1, 4)
            f = _np.zeros(len(l))  # f(n) of different given box length n
            # 2>3 xrange>range
            for i in range(0, len(l)):
                n = int(l[i])  # for each box length l[i]
                # 2>3 xrange>range
                for j in range(0, len(x), n):  # for each box
                    if j + n < len(x):
                        c = range(j, j + n)
                        c = _np.vstack([c, _np.ones(n)]).T  # coordinates of time in the box
                        z = y[j:j + n]  # the value of example_data in the box
                        f[i] += _np.linalg.lstsq(c, z)[1]  # add residue in this box
                f[i] /= ((len(x) / n) * n)
            f = _np.sqrt(f)

            return _np.linalg.lstsq(_np.vstack([_np.log(l), _np.ones(len(l))]).T, _np.log(f))[0][0]
