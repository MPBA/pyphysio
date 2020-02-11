# coding=utf-8
from __future__ import division
import numpy as _np
import scipy.stats as _stats
from scipy.signal import gaussian as _gaussian, filtfilt as _filtfilt, filter_design as _filter_design, \
    deconvolve as _deconvolve, firwin as _firwin, convolve as _convolve
from matplotlib.pyplot import plot as _plot
from ..BaseFilter import Filter as _Filter
from ..Signal import EvenlySignal as _EvenlySignal, UnevenlySignal as _UnevenlySignal
from ..Utility import abstractmethod as _abstract
from ..tools.Tools import SignalRange
from collections import Sequence
__author__ = 'AleB'


class Normalize(_Filter):
    """
    Normalized the input signal using the general formula: ( signal - BIAS ) / RANGE

    Parameters
    -------------------
    norm_method : 
        Method for the normalization. Available methods are:
    * 'mean' - remove the mean [ BIAS = mean(signal); RANGE = 1 ]
    * 'standard' - standardization [ BIAS = mean(signal); RANGE = std(signal) ]
    * 'min' - remove the minimum [ BIAS = min(signal); RANGE = 1 ]
    * 'maxmin' - maxmin normalization [ BIAS = min(signal); RANGE = ( max(signal) - min(signal ) ]
    * 'custom' - custom, bias and range are manually defined [ BIAS = bias, RANGE = range ]
    
    norm_bias : float, default = 0
        Bias for custom normalization
    norm_range : float, !=0, default = 1
        Range for custom normalization

    Returns
    -------
    signal: 
        The normalized signal. 

    """

    def __init__(self, norm_method='standard', norm_bias=0, norm_range=1):
        assert norm_method in ['mean', 'standard', 'min', 'maxmin', 'custom'],\
            "norm_method must be one of 'mean', 'standard', 'min', 'maxmin', 'custom'"
        if norm_method == "custom":
            assert norm_range != 0, "norm_range must not be zero"
        _Filter.__init__(self, norm_method=norm_method, norm_bias=norm_bias, norm_range=norm_range)

    @classmethod
    def algorithm(cls, signal, params):
        from ..indicators.TimeDomain import Mean as _Mean, StDev as _StDev

        method = params['norm_method']
        if method == "mean":
            return signal - _Mean()(signal)
        elif method == "standard":
            return (signal - _Mean()(signal)) / _StDev()(signal)
        elif method == "min":
            return signal - _np.min(signal)
        elif method == "maxmin":
            return (signal - _np.min(signal)) / (_np.max(signal) - _np.min(signal))
        elif method == "custom":
            return (signal - params['norm_bias']) / params['norm_range']



class IIRFilter(_Filter):
    """
    Filter the input signal using an Infinite Impulse Response filter.

    Parameters
    ----------
    fp : list or float
        The pass frequencies
    fs : list or float
        The stop frequencies
    
    Optional parameters
    -------------------
    loss : float, >0, default = 0.1
        Loss tolerance in the pass band
    att : float, >0, default = 40
        Minimum attenuation required in the stop band.
    ftype : str, default = 'butter'
        Type of filter. Available types: 'butter', 'cheby1', 'cheby2', 'ellip', 'bessel'

    Returns
    -------
    signal : EvenlySignal
        Filtered signal

    Notes
    -----
    This is a wrapper of *scipy.signal.filter_design.iirdesign*. Refer to `scipy.signal.filter_design.iirdesign`
    for additional information
    """

    def __init__(self, fp, fs, loss=.1, att=40, ftype='butter'):
        assert loss > 0, "Loss value should be positive"
        assert att > 0, "Attenuation value should be positive"
        assert att > loss, "Attenuation value should be greater than loss value"
        assert ftype in ['butter', 'cheby1', 'cheby2', 'ellip', 'bessel'],\
            "Filter type must be in ['butter', 'cheby1', 'cheby2', 'ellip', 'bessel']"
        _Filter.__init__(self, fp=fp, fs=fs, loss=loss, att=att, ftype=ftype)

    @classmethod
    def algorithm(cls, signal, params):
        fsamp = signal.get_sampling_freq()
        fp, fs, loss, att, ftype = params["fp"], params["fs"], params["loss"], params["att"], params["ftype"]

        if isinstance(signal, _UnevenlySignal):
            cls.warn('Filtering Unevenly signal is undefined. Returning original signal.')
            return signal

        nyq = 0.5 * fsamp
        fp = _np.array(fp)
        fs = _np.array(fs)

        wp = fp / nyq
        ws = fs / nyq
        # noinspection PyTupleAssignmentBalance
        b, a = _filter_design.iirdesign(wp, ws, loss, att, ftype=ftype, output="ba")

        sig_filtered = signal.clone_properties(_filtfilt(b, a, signal.get_values()))

        if _np.isnan(sig_filtered[0]):
            cls.warn('Filter parameters allow no solution. Returning original signal.')
            return signal
        else:
            return sig_filtered

    @_abstract
    def plot(self):
        pass

class FIRFilter(_Filter):
    """
    Filter the input signal using a Finite Impulse Response filter.

    Parameters
    ----------
    fp : list or float
        The pass frequencies
    fs : list or float
        The stop frequencies
    
    Optional parameters
    -------------------
    loss : float, >0, default = 0.1
        Loss tolerance in the pass band
    att : float, >0, default = 40
        Minimum attenuation required in the stop band.
    wtype : str, default = 'hamming'
        Type of filter. Available types: 'hamming'

    Returns
    -------
    signal : EvenlySignal
        Filtered signal

    Notes
    -----
    This is a wrapper of *scipy.signal.firwin*. Refer to `scipy.signal.firwin`
    for additional information
    """

    def __init__(self, fp, fs, loss=0.1, att=40, wtype='hamming'):
        assert loss > 0, "Loss value should be positive"
        assert att > 0, "Attenuation value should be positive"
        assert att > loss, "Attenuation value should be greater than loss value"
        assert wtype in ['hamming'],\
            "Window type must be in ['hamming']"
        _Filter.__init__(self, fp=fp, fs=fs, loss=loss, att=att, wtype=wtype)

    @classmethod
    def algorithm(cls, signal, params):
        fsamp = signal.get_sampling_freq()
        fp, fs, loss, att, wtype = params["fp"], params["fs"], params["loss"], params["att"], params["wtype"]

        if isinstance(signal, _UnevenlySignal):
            cls.warn('Filtering Unevenly signal is undefined. Returning original signal.')
            return signal

        fp = _np.array(fp)
        fs = _np.array(fs)
        
        if att>0:
            att = -att
        d1 = 10**(loss/10)
        d2 = 10**(att/10)
        Dsamp = _np.min(abs(fs-fp))/fsamp
        

        # from https://dsp.stackexchange.com/questions/31066/how-many-taps-does-an-fir-filter-need
        N = int(2/3*_np.log10(1/(10*d1*d2))*fsamp/Dsamp)
                
        pass_zero=True
                  
        if isinstance(fp, Sequence):
            if fp[0]>fs[0]:
                pass_zero=False
        else:    
            if fp[0]>fs[0]:
                pass_zero=False
        
            
        
        nyq = 0.5 * fsamp
        fp = _np.array(fp)
        wp = fp / nyq
        
        if N%2 ==0:
            N+=1
        b = _firwin(N, wp, width=Dsamp, window=wtype, pass_zero=pass_zero)
        sig_filtered = signal.clone_properties(_convolve(signal.get_values(), b, mode='same'))

        if _np.isnan(sig_filtered[0]):
            cls.warn('Filter parameters allow no solution. Returning original signal.')
            return signal
        else:
            return sig_filtered

    @_abstract
    def plot(self):
        pass

class KalmanFilter(_Filter):
    def __init__(self, R, ratio=1, win_len=1, win_step=0.5):
        assert R > 0, "R should be positive"
        if ratio is not None:
            assert ratio > 1, "ratio should be >1"
        assert win_len > 0, "Window length value should be positive"
        assert win_step > 0, "Window step value should be positive"
        
        _Filter.__init__(self, R=R, ratio=ratio, win_len=win_len, win_step=win_step)
        
    @classmethod
    def algorithm(cls, signal, params):
        R = params['R']
        ratio = params['ratio']
        win_len = params['win_len']
        win_step = params['win_step']
        
        sz = len(signal)
        
        rr = SignalRange(win_len, win_step)(signal)
        Q = _np.nanmedian(rr)/ratio
            
        P = 1
        
        x_out = signal.get_values().copy()
        for k in range(1,sz):
                x_ = x_out[k-1]
                P_ = P + Q
            
                # measurement update
                K = P_ / (P_ + R)
                x_out[k] = x_ + K * (x_out[k] - x_)
                P = (1 - K ) * P_

        x_out = signal.clone_properties(x_out)
        return(x_out)

############
class ImputeNAN(_Filter):
    def __init__(self, win_len=5, allnan='nan'):
        assert win_len>0, "win_len should be >0"
        assert allnan in ['zeros', 'nan']
        _Filter.__init__(self, win_len = win_len, allnan=allnan)
        
    @classmethod
    def algorithm(cls, signal, params):
        def group_consecutives(vals, step=1):
            """Return list of consecutive lists of numbers from vals (number list)."""
            run = []
            result = [run]
            expect = None
            for v in vals:
                if (v == expect) or (expect is None):
                    run.append(v)
                else:
                    run = [v]
                    result.append(run)
                expect = v + step
            return result

        #%
        win_len = params['win_len']*signal.get_sampling_freq()
        allnan = params['allnan']
        
        s = signal.get_values().copy()
        if _np.isnan(s).all():
            if allnan == 'nan':
                return(signal)
            else:
                s = _np.zeros_like(s)
                s_out = signal.clone_properties(s)
                return(s_out)
        
        idx_nan = _np.where(_np.isnan(s))[0]
        segments = group_consecutives(idx_nan)

        #%
        if len(segments[0])>=1:
            for i_seg, SEG in enumerate(segments):
                idx_st = SEG[0]
                idx_sp = SEG[-1]
                idx_win_pre = _np.arange(-int(win_len/2), 0, 1)+idx_st
                idx_win_pre = idx_win_pre[_np.where(idx_win_pre>0)[0]] #not before signal start

                STD = []
                if len(idx_win_pre)>=3:
                    STD.append(_np.nanstd(s[idx_win_pre]))
                
                idx_win_post = _np.arange(0, int(win_len/2))+idx_sp+1
                idx_win_post = idx_win_post[_np.where(idx_win_post<len(s))[0]]
                
                if len(idx_win_post)>=3:
                    STD.append(_np.nanstd(s[idx_win_post]))
                
                if len(STD)>0 and not (_np.isnan(STD).all()):
                    STD = _np.nanmin(STD)
                else:
                    STD = 0
                    
                idx_win = _np.hstack([idx_win_pre, idx_win_post]).astype(int)
                idx_win = idx_win[_np.where(~_np.isnan(s[idx_win]))[0]] # remove nans
                
                if len(idx_win)>3:
                    R = _stats.linregress(idx_win, s[idx_win])
                    s_nan = _np.array(SEG)*R[0]+R[1] + _np.random.normal(scale=STD, size = len(SEG))
                else:
                    s_nan = _np.nanmean(s)*_np.ones(len(SEG))
                s[SEG] = s_nan
        
        signal_out = signal.clone_properties(s)
        return(signal_out)


class RemoveSpikes(_Filter):
    def __init__(self, K=2, N=1, dilate=0, D=0.95, method='step'):
        assert K > 0, "K should be positive"
        assert isinstance(N, int) and N>0, "N value not valid"
        assert dilate>=0, "dilate should be >= 0.0"
        assert D>=0, "D should be >= 0.0"
        assert method in ['linear', 'step']
        _Filter.__init__(self, K=K, N=N, dilate=dilate, D=D, method=method)
    
    @classmethod
    def algorithm(cls, signal, params):
        K = params['K']
        N = params['N']
        dilate = params['dilate']
        D = params['D']
        method = params['method']
        fs = signal.get_sampling_freq()
        
        sig_diff = abs(signal[N:] - signal[:-N])
        ds_mean = _np.nanmean(sig_diff)
        
        idx_spikes = _np.where(sig_diff>K*ds_mean)[0]+N//2
        spikes = _np.zeros(len(signal))
        spikes[idx_spikes] = 1
        win = _np.ones(1+int(2*dilate*fs))
        spikes = _np.convolve(spikes, win, 'same')
        idx_spikes = _np.where(spikes>0)[0]
        
        x_out = signal.get_values().copy()
        
        #TODO add linear connector method
        if method == 'linear':
            diff_idx_spikes = _np.diff(idx_spikes)
            new_spike = _np.where(diff_idx_spikes > 1)[0] + 1
            new_spike = _np.r_[0, new_spike, -1]
            for I in range(len(new_spike)-1):
                IDX_START = idx_spikes[new_spike[I]] -1
                IDX_STOP = idx_spikes[new_spike[I+1]-1] +1
                
                L = IDX_STOP - IDX_START + 1
                x_start = x_out[IDX_START]
                x_stop = x_out[IDX_STOP]
                coefficient = (x_stop - x_start)/ L
                
                x_out[IDX_START:IDX_STOP+1] = coefficient*_np.arange(L) + x_start
        else:
                
            for IDX in idx_spikes:
                delta = x_out[IDX] - x_out[IDX-1]
                x_out[IDX:] = x_out[IDX:] - D*delta
        x_out = signal.clone_properties(x_out)
        return(x_out)

class DenoiseEDA(_Filter):
    """
    Remove noise due to sensor displacement from the EDA signal.
    
    Parameters
    ----------
    threshold : float, >0
        Threshold to detect the noise
        
    Optional parameters
    -------------------
    
    win_len : float, >0, default = 2
        Length of the window
   
    Returns
    -------
    signal : EvenlySignal
        De-noised signal
            
    """

    def __init__(self, threshold, win_len=2):
        assert threshold > 0, "Threshold value should be positive"
        assert win_len > 0, "Window length value should be positive"
        _Filter.__init__(self, threshold=threshold, win_len=win_len)

    @classmethod
    def algorithm(cls, signal, params):
        threshold = params['threshold']
        win_len = params['win_len']

        # remove fluctiations
        noise = ConvolutionalFilter(irftype='triang', win_len=win_len, normalize=True)(abs(_np.diff(signal)))

        # identify noisy portions
        idx_ok = _np.where(noise <= threshold)[0]

        # fix start and stop of the signal for the following interpolation
        if idx_ok[0] != 0:
            idx_ok = _np.r_[0, idx_ok].astype(int)

        if idx_ok[-1] != len(signal) - 1:
            idx_ok = _np.r_[idx_ok, len(signal) - 1].astype(int)

        denoised = _UnevenlySignal(signal[idx_ok], signal.get_sampling_freq(), x_values=idx_ok, x_type='indices',
                                   duration=signal.get_duration())

        # interpolation
        signal_out = denoised.to_evenly('linear')
        return signal_out


class ConvolutionalFilter(_Filter):
    """
    Filter a signal by convolution with a given impulse response function (IRF).

    Parameters
    ----------
    irftype : str
        Type of IRF to be generated. 'gauss', 'rect', 'triang', 'dgauss', 'custom'.
    win_len : float, >0 (> 8/fsamp for 'gaussian')
        Duration of the generated IRF in seconds (if irftype is not 'custom')
    
    Optional parameters
    -------------------
    irf : numpy.array
        IRF to be used if irftype is 'custom'
    normalize : boolean, default = True
        Whether to normalizes the IRF to have unitary area
    
    Returns
    -------
    signal : EvenlySignal
        Filtered signal

    """

    def __init__(self, irftype, win_len=0, irf=None, normalize=True):
        assert irftype in ['gauss', 'rect', 'triang', 'dgauss', 'custom'],\
            "IRF type must be in ['gauss', 'rect', 'triang', 'dgauss', 'custom']"
        assert irftype == 'custom' or win_len > 0, "Window length value should be positive"
        _Filter.__init__(self, irftype=irftype, win_len=win_len, irf=irf, normalize=normalize)

    # TODO (Andrea): TEST normalization and results
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
                    if n < 8:
                        # TODO (Andrea): test, sometimes it returns nan
                        cls.error(
                            "'win_len' too short to generate a gaussian IRF, expected > " + str(_np.ceil(8 / fsamp)))
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

        signal_out = signal.clone_properties(signal_f[n:-n])
        return signal_out

    @classmethod
    def plot(cls):
        pass


class DeConvolutionalFilter(_Filter):
    """
    Filter a signal by deconvolution with a given impulse response function (IRF).

    Parameters
    ----------
    irf : numpy.array
        IRF used to deconvolve the signal
    
    Optional parameters
    -------------------
    
    normalize : boolean, default = True
        Whether to normalize the IRF to have unitary area
    deconv_method : str, default = 'sps'
        Available methods: 'fft', 'sps'. 'fft' uses the fourier transform, 'sps' uses the scipy.signal.deconvolve
         function
        
    Returns
    -------
    signal : EvenlySignal
        Filtered signal

    """

    def __init__(self, irf, normalize=True, deconv_method='sps'):
        # TODO (Andrea): "check that irf[0]>0 to avoid scipy BUG" is it normal? Need to put a check?
        assert deconv_method in ['fft', 'sps'], "Deconvolution method not valid"
        _Filter.__init__(self, irf=irf, normalize=normalize, deconv_method=deconv_method)

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
            cls.warn('sps based deconvolution needs to be tested. Use carefully.')
            out, _ = _deconvolve(signal, irf)
        else:
            cls.error('Deconvolution method not implemented. Returning original signal.')
            out = signal.get_values()

        out_signal = signal.clone_properties(abs(out))

        return out_signal

    def plot(self):
        _plot(self._params['irf'])
