# coding=utf-8
from __future__ import division

from ..BaseIndicator import Indicator as _Indicator
from ..tools.Tools import PSD as PSD
import numpy as _np

__author__ = 'AleB'


class InBand(_Indicator):
    """
    Extract the PSD of a given frequency band
    

    Parameters
    ----------
    freq_min : float, >0
        Left bound of the frequency band
    freq_max : float, >0
        Right bound of the frequency band
    method : 'ar', 'welch' or 'fft'
        Method to estimate the PSD
        
    Additional parameters
    ---------------------
    For the PSD (see pyphysio.tools.Tools.PSD), for instance:
        
    interp_freq : float, >0
        Frequency used to (re-)interpolate the signal

    Returns
    -------
    freq : numpy array
        Frequencies in the frequency band
    psd : float
        Power Spectrum Density in the frequency band
    """

    def __init__(self, freq_min, freq_max, method, **kwargs):
        _Indicator.__init__(self, freq_min=freq_min, freq_max=freq_max, method=method, **kwargs)

    @classmethod
    def algorithm(cls, data, params):
        freq, spec = PSD(**params)(data)
        # freq is sorted so
        i_min = _np.searchsorted(freq, params["freq_min"])
        i_max = _np.searchsorted(freq, params["freq_max"])
        return freq[i_min:i_max], spec[i_min:i_max]


class PowerInBand(_Indicator):
    """
    Estimate the power in given frequency band

    Parameters
    ----------
    freq_min : float, >0
        Left bound of the frequency band
    freq_max : float, >0
        Right bound of the frequency band
    method : 'ar', 'welch' or 'fft'
        Method to estimate the PSD
        
    Additional parameters
    ---------------------
    For the PSD (see pyphysio.tools.Tools.PSD):
        
    interp_freq : float, >0
        Frequency used to (re-)interpolate the signal

    Returns
    -------
    power : float
        Power in the frequency band
    """

    def __init__(self, freq_min, freq_max, method, **kwargs):
        _Indicator.__init__(self, freq_min=freq_min, freq_max=freq_max, method=method, **kwargs)

    @classmethod
    def algorithm(cls, data, params):
        freq, powers = InBand(**params)(data)
        return _np.sum(powers)


class PeakInBand(_Indicator):
    """
    Estimate the peak frequency in a given frequency band

    Parameters
    ----------
    freq_min : float, >0
        Left bound of the frequency band
    freq_max : float, >0
        Right bound of the frequency band
    method : 'ar', 'welch' or 'fft'
        Method to estimate the PSD
        
    Additional parameters
    ---------------------
    For the PSD (see pyphysio.tools.Tools.PSD):
        
    interp_freq : float, >0
        Frequency used to (re-)interpolate the signal

    Returns
    -------
    peak : float
        Peak frequency
    """

    def __init__(self, freq_min, freq_max, method, **kwargs):
        _Indicator.__init__(self, freq_min=freq_min, freq_max=freq_max, method=method, **kwargs)
    
    @classmethod
    def algorithm(cls, data, params):
        freq, power = InBand(**params)(data)
        return freq[_np.argmax(power)]

