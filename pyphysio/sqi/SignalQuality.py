# coding=utf-8
import numpy as _np

from ..BaseIndicator import Indicator as _Indicator
from ..indicators.FrequencyDomain import PowerInBand as _PowerInBand
import scipy.stats as _sps
from ..filters.Filters import ImputeNAN as _ImputeNAN

__author__ = 'AleB'

class Kurtosis(_Indicator):
    """
    Compute the Kurtosis of the signal
    
    """
    def __init__(self, **kwargs):
        _Indicator.__init__(self, **kwargs)

    @classmethod
    def algorithm(cls, data, params):
        k = _sps.kurtosis(data.get_values())
        return(k)

class Entropy(_Indicator):
    def __init__(self, nbins=25, **kwargs):
        _Indicator.__init__(self, nbins=nbins, **kwargs)
    
    @classmethod
    def algorithm(cls, data, params):
        if _np.isnan(data).all():
            return(_np.nan)
        nbins=params['nbins']
        p_data = _np.histogram(data.get_values(), bins=nbins)[0]/len(data) # calculates the probabilities
        entropy = _sps.entropy(p_data)  # input probabilities to get the entropy 
        return(entropy)

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
    def __init__(self, method='ar', bandN=[5,14], bandD=[5,50],**kwargs):
        _Indicator.__init__(self, method=method, bandN=bandN, bandD=bandD, **kwargs)

    @classmethod
    def algorithm(cls, data, params):
        bandN = params['bandN']
        bandD = params['bandD']
        assert bandD[1] < data.get_sampling_freq()/2, 'The higher frequency in bandD is greater than fsamp/2: cannot compute power' # CHECK: check sampling frequency of the signal (e.g. <=128)
        p_N = _PowerInBand(bandN[0], bandN[1], params['method'])(data)
        p_D = _PowerInBand(bandD[0],bandD[1], params['method'])(data)
        return(p_N/p_D)

class CVSignal(_Indicator):
    """
    Compute the Coefficient of variation of the signal
    
    See: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3859838/

    """
    def __init__(self, **kwargs):
        _Indicator.__init__(self, **kwargs)

    @classmethod
    def algorithm(cls, data, params):
        mean = _np.nanmean(data)
        sd = _np.nanstd(data)
        cv = sd/mean
        return(cv)

class PercentageNAN(_Indicator):
    """
    Compute the Percentage of NaNs

    """
    def __init__(self, **kwargs):
        _Indicator.__init__(self, **kwargs)

    @classmethod
    def algorithm(cls, data, params):
        n_nan = _np.sum(_np.isnan(data))
        data = _ImputeNAN()(data)
        return(100*n_nan/len(data))
