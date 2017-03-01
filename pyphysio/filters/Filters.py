# coding=utf-8
from __future__ import division
import numpy as _np
from scipy.signal import gaussian as _gaussian, filtfilt as _filtfilt, filter_design as _filter_design, deconvolve as _deconvolve
from ..BaseFilter import Filter as _Filter
from ..Signal import EvenlySignal as _EvenlySignal, UnevenlySignal as _UnevenlySignal, Signal as _Signal
from ..Utility import PhUI as _PhUI
from ..Parameters import Parameter as _Par
from ..Utility import abstractmethod as _abstract

__author__ = 'AleB'

"""
Filters are processing steps that take as input a SIGNAL and gives as output another SIGNAL of the SAME NATURE.
"""


class Normalize(_Filter):
    """
    Normalized the input signal using the general formula: ( signal - BIAS ) / RANGE

    Parameters
    ----------
    norm_method : str, default = 'standard'
        Method for the normalization. Available methods are:
        'mean' - remove the mean [ BIAS = mean(signal); RANGE = 1 ]
        'standard' - standardization [ BIAS = mean(signal); RANGE = std(signal) ]
        'min' - remove the minimum [ BIAS = min(signal); RANGE = 1 ]
        'maxmin' - maxmin normalization [ BIAS = min(signal); RANGE = ( max(signal) - min(signal ) ]
        'custom' - custom, bias and range are manually defined [ BIAS = bias, RANGE = range ]

    bias : float, default = 0
        Bias for custom normalization
    range : float, !=0, default = 1
        Range for custom normalization

    Returns
    -------
    signal: 
        The normalized signal. 

    Notes
    -----
        ...
    """
    _params_descriptors = {
        'norm_method': _Par(0, str, 'Method for the normalization.', 'standard', lambda x: x in ['mean', 'standard', 'min', 'maxmin', 'custom']),
        'norm_bias': _Par(2, float, 'Bias for custom normalization', activation=lambda x, p: p['norm_method'] == 'custom'),
        'norm_range': _Par(2, float, 'Range for custom normalization', activation=lambda x, p: p['norm_method'] == 'custom')
        #TODO: check norm_range !=0
    }
    
    class Types(object):
        Mean = 'mean'
        MeanSd = 'standard'
        Min = 'min'
        MaxMin = 'maxmin'
        Custom = 'custom'

    @classmethod
    def algorithm(cls, signal, params):
        from ..indicators.TimeDomain import Mean as _Mean, StDev as _StDev

        method = params['norm_method']
        if method == Normalize.Types.Mean:
            return signal - _Mean()(signal)
        elif method == Normalize.Types.MeanSd:
            return (signal - _Mean()(signal)) / _StDev()(signal)
        elif method == Normalize.Types.Min:
            return signal - _np.min(signal)
        elif method == Normalize.Types.MaxMin:
            return (signal - _np.min(signal)) / (_np.max(signal) - _np.min(signal))
        elif method == Normalize.Types.Custom:
            return (signal - params['norm_bias']) / params['norm_range']


class Diff(_Filter):
    """
    Computes the differences between adjacent samples.

    Parameters
    ----------
    
    Optional:
    degree : int, >0, default = 1
        Sample interval to compute the differences
    
    Returns
    -------
    signal : 
        Differences signal. 

    Notes
    -----
    Note that the length of the returned signal is the lenght of the input_signal minus degree.
    """
    
    _params_descriptors = {
        'degree': _Par(0, int, 'Degree of the differences', 1, lambda x: x>0)
    }
    
    @classmethod
    def algorithm(cls, signal, params):
        """
        Calculates the differences between consecutive values
        """
#        if isinstance(signal, _Signal) and not isinstance(signal, _EvenlySignal):
#            cls.log("Computing %s on '%s' may not make sense." % (cls.__name__, signal.__class__.__name__))
        degree = params['degree']

        sig_1 = signal[:-degree]
        sig_2 = signal[degree:]
        
        # TODO: should return the same signal type of the input, with same characteristics
        out = _EvenlySignal(values=sig_2 - sig_1,
                            sampling_freq=signal.get_sampling_freq(),
                            signal_nature=signal.get_signal_nature(),
                            start_time=signal.get_start_time()+ degree / signal.get_sampling_freq())

        return out


class IIRFilter(_Filter):
    """
    Filter the input signal using an Infinite Impulse Response filter.

    Parameters
    ----------
    fp : list
        The pass frequencies
    fs : list
        The stop frequencies
    
    Optional:
    loss : float, >0, default = 0.1
        Loss tolerance in the pass band
    att : float, >0, default = 40
        Minimum attenuation required in the stop band.
    ftype : str, default = 'butter'
        Type of filter. Available types: 'butter', 'cheby1', 'cheby2', 'ellip', 'bessel'

    Returns
    -------
    signal: EvenlySignal
        Filtered signal

    Notes
    -----
    This is a wrapper of `scipy.signal.filter_design.iirdesign`. Refer to `scipy.signal.filter_design.iirdesign` for additional information
    """
    
    _params_descriptors = {
        'fp': _Par(2, list, 'The pass frequencies'),
        'fs': _Par(2, list, 'The stop frequencies'),
        'loss': _Par(0, float, 'Loss tolerance in the pass band', 0.1, lambda x: x > 0),
        'att': _Par(0, float, 'Minimum attenuation required in the stop band.', 40, lambda x: x > 0),
        'ftype': _Par(0, str, 'Type of filter', 'butter', lambda x: x in ['butter', 'cheby1', 'cheby2', 'ellip', 'bessel'])
    }
    
    @classmethod
    def algorithm(cls, signal, params):
        fsamp = signal.get_sampling_freq()
        fp, fs, loss, att, ftype = params["fp"], params["fs"], params["loss"], params["att"], params["ftype"]

        if isinstance(signal, _Signal) and not isinstance(signal, _EvenlySignal):
            cls.warn(cls.__name__ + ': Filtering Unevenly signal is undefined. Returning original signal.')
        
        # TODO (feature): check that fs and fp are meaningful
        # TODO (feature): if A and B already exist and fsamp is not changed skip the following
        # TODO (feature): check if fs, fp, fsamp allow no solution for the filter
        nyq = 0.5 * fsamp
        fp = _np.array(fp)
        fs = _np.array(fs)
        
        wp = fp / nyq
        ws = fs / nyq
        b, a = _filter_design.iirdesign(wp, ws, loss, att, ftype=ftype, output="ba")
                
        sig_filtered = _EvenlySignal(_filtfilt(b, a, signal.get_values()), signal.get_sampling_freq(), signal.get_signal_nature(), signal.get_start_time())
        if _np.isnan(sig_filtered[0]):
            cls.warn(cls.__name__ + ': Filter parameters allow no solution. Returning original signal.')
            return signal
        else:
            return sig_filtered

    @_abstract
    def plot(self):
        # TODO (feature): plot frequency response
        pass

class DenoiseEDA(_Filter):
    """
    Removes noise due to sensor displacement from the EDA signal.
    
    Parameters
    ----------
    threshold : float, >0
        Threshold to detect the noise
        
    Optional:
    
    win_len : float, >0, default=2
        Length of the window
   
    Returns
    -------
    signal : EvenlySignal
        De-noised signal
            
    Notes
    -----
    See REF for more infos.
    #TODO: insert ref
    """
    
    _params_descriptors = {
        'threshold': _Par(2, float, 'Threshold to detect the noise', constraint=lambda x: x > 0),
        'win_len': _Par(0, float, 'Length of the window', 2, lambda x: x > 0)
    }
    
    @classmethod
    def algorithm(cls, signal, params):
        
        threshold = params['threshold']
        win_len = params['win_len']
        
        #TODO (feature): detect the long periods of drop
        #remove fluctiations
        noise = ConvolutionalFilter(irftype ='triang', win_len = win_len, normalize=True)(abs(_np.diff(signal)))
        
        #identify noisy portions        
        idx_ok = _np.where(noise <= threshold)[0]
        
        #fix start and stop of the signal for the followinf interpolation
        if idx_ok[0] != 0:
            idx_ok = _np.r_[0, idx_ok].astype(int)
        
        if idx_ok[-1] != len(signal)-1:
            idx_ok = _np.r_[idx_ok, len(signal)-1].astype(int)

        denoised = _UnevenlySignal(signal[idx_ok], signal.get_sampling_freq(), x_values = idx_ok, x_type = 'indices')
        
        #interpolation
        signal_out = denoised.to_evenly('linear')
        return signal_out

class ConvolutionalFilter(_Filter):
    """
    Filter a signal by convolution with a given impulse response function (IRF).

    Parameters
    ----------
    irftype : str
        Type of IRF to be generated. 'gauss', 'rect', 'triang', 'dgauss', 'custom'.
    win_len : float, >0 (>8 for 'gaussian")
        Durarion of the generated IRF in seconds (if irftype is not 'custom')
    irf : numpy.array
        IRF to be used if irftype is 'custom'
    
    Optional:
    normalize : boolean, default = True
        Whether to normalizes the IRF to have unitary area
    
    Returns
    -------
    signal : EvenlySignal
        Filtered signal

    """
    
    _params_descriptors = {
        'irftype': _Par(2, str, 'Type of IRF to be generated.', constraint=lambda x: x in ['gauss', 'rect', 'triang', 'dgauss', 'custom']), 
        'win_len': _Par(2, float, "Durarion of the generated IRF in seconds (if irftype is not 'custom')", constraint=lambda x: x > 0, activation=lambda x, p: p['irftype'] != 'custom'),        
        'irf': _Par(2, list, "IRF to be used if irftype is 'custom'", activation=lambda x, p: p['irftype'] == 'custom'),
        'normalize': _Par(1, bool, 'Whether to normalizes the IRF to have unitary area', True)
    }
    
#    class Types(object):
#        Same = 'none'
#        Gauss = 'gauss'
#        Rect = 'rect'
#        Triang = 'triang'
#        Dgauss = 'dgauss'
#        Custom = 'custom'
    
    #TODO: TEST normalization and results
    @classmethod
    def algorithm(cls, signal, params):
        irftype = params["irftype"]
        normalize = params["normalize"]

        fsamp = signal.get_sampling_freq()
        irf = None

        if irftype == 'custom':
            if 'irf' not in params:
                cls.error("'irf' parameter missing.")
                return signal
            else:
                irf = _np.array(params["irf"])
                n = len(irf)
        else:
            if 'win_len' not in params:
                cls.error("'win_len' parameter missing.")
                return signal
            else:
                n = int(params['win_len'] * fsamp)

                if irftype == 'gauss':
                    if n<8:
                        #TODO: test, sometimes it returns nan
                        cls.error("'win_len' too short to generate a gaussian IRF, expected > "+str(_np.ceil(8/fsamp)))
                    std = _np.floor(n / 8) 
                    irf = _gaussian(n, std)
                elif irftype == 'rect':
                    irf = _np.ones(n)
                elif irftype == 'triang':
                    irf_1 = _np.arange(n // 2)
                    irf_2 = irf_1[-1] - _np.arange(n // 2)
                    if n % 2 == 0:
                        irf = _np.r_[irf_1, irf_2]
                    else:
                        irf = _np.r_[irf_1, irf_1[-1] + 1, irf_2]
                elif irftype == 'dgauss':
                    std = _np.round(n / 8)
                    g = _gaussian(n, std)
                    irf = _np.diff(g)
                
        # NORMALIZE
        if normalize:
            irf = irf / _np.sum(irf)

        signal_ = _np.r_[_np.ones(n) * signal[0], signal, _np.ones(n) * signal[-1]]  # TESTME

        signal_f = _np.convolve(signal_, irf, mode='same')

        signal_out = _EvenlySignal(signal_f[n:-n], signal.get_sampling_freq(), signal.get_signal_nature(),
                                   signal.get_start_time())
        return signal_out

    @classmethod
    def plot(cls):
        # TODO (feature): plot the IRF
        pass


class DeConvolutionalFilter(_Filter):
    """
    Filter a signal by deconvolution with a given impulse response function (IRF).

    Parameters
    ----------
    irf : numpy.array
        IRF used to deconvolve the signal
    
    Optional:    
    normalize : boolean, default = True
        Whether to normalize the IRF to have unitary area
    method : str
        Available methods: 'fft', 'sps'. 'fft' uses the fourier transform, 'sps' uses the scipy.signal.deconvolve function
        
    Returns
    -------
    signal : EvenlySignal
        Filtered signal

    """

    _params_descriptors = {
        'irf': _Par(2, list, 'IRF used to deconvolve the signal'),#TODO: check that irf[0]>0 to avoid scipy BUG
        'normalize': _Par(0, bool, 'Whether to normalize the IRF to have unitary area', True),
        'deconv_method': _Par(0, str, 'Deconvolution method.', 'fft', lambda x: x in ['fft', 'sps'])
    }
    
    @classmethod
    def algorithm(cls, signal, params):
        irf = params["irf"]
        normalize = params["normalize"]
        deconvolution_method = params["deconv_method"]
        
        if normalize:
            irf = irf / _np.sum(irf)
        if deconvolution_method == 'fft':
            l = len(signal)
            fft_signal = _np.fft.fft(signal, n=l)
            fft_irf = _np.fft.fft(irf, n=l)
            out = _np.fft.ifft(fft_signal / fft_irf)
        elif deconvolution_method == 'sps':
            cls.warn(cls.__name__+': sps based deconvolution needs to be tested. Use carefully.')
            out, _  = _deconvolve(signal, irf)
        else:
            print('Deconvolution method not implemented. Returning original signal.')
            out = signal.get_values()
                
            
        out_signal = _EvenlySignal(abs(out), signal.get_sampling_freq(), signal.get_signal_nature(), signal.get_start_time())

        return out_signal

    @classmethod
    def plot(cls):
        # TODO (new feature): plot the irf
        pass