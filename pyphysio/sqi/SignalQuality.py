# coding=utf-8
import numpy as _np

from ..BaseIndicator import Indicator as _Indicator
from ..indicators.FrequencyDomain import PowerInBand
import scipy.stats as sps

__author__ = 'AleB'

class Kurtosis(_Indicator):
    """
    Compute the Kurtosis of the signal
    
    """
    def __init__(self, **kwargs):
        _Indicator.__init__(self, **kwargs)

    @classmethod
    def algorithm(cls, data, params):
        k = sps.kurtosis(data.get_values())
        return(k)

class DerivativeEnergy(_Indicator):
    """
    Compute the Derivative Energy

    """
    def __init__(self, **kwargs):
        _Indicator.__init__(self, **kwargs)
    
    @classmethod
    def algorithm(cls, data, params):
        x = data.get_values()
        de = _np.sqrt(_np.nanmean(_np.power(_np.diff(x), 2)))
        return(de)
        
class SpectralPowerRatio(_Indicator):
    """
    Compute the Spectral Power Ratio

    """
    def __init__(self, method='ar', **kwargs):
        _Indicator.__init__(self, method=method, **kwargs)

    @classmethod
    def algorithm(cls, data, params):
        # TODO: check sampling frequency of the signal (e.g. <=128)
        p_5_14 = PowerInBand(5,14, params['method'])(data)
        p_5_50 = PowerInBand(5,50, params['method'])(data)
        return(p_5_14/p_5_50)