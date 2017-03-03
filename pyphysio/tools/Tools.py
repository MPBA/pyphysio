# coding=utf-8
from __future__ import division
import numpy as _np
import sys as _sys
from scipy.signal import welch as _welch
import scipy.optimize as _opt
from spectrum import aryule as _aryule, arma2psd as _arma2psd, AIC as _AIC
import itertools as _itertools
from ..BaseTool import Tool as _Tool

from ..Signal import UnevenlySignal as _UnevenlySignal, EvenlySignal as _EvenlySignal
from ..filters.Filters import Diff as _Diff, ConvolutionalFilter as _ConvFlt
from ..Utility import PhUI as _PhUI
from ..Parameters import Parameter as _Par


class PeakDetection(_Tool):
    """
    Estimate the maxima and the minima in the signal (in particular for periodic signals).

    Parameters
    ----------
    delta : float or list
        Threshold for the detection of the peaks. If it is a list it must have the same length of the signal.
        
    Optional:
    refractory : float, >=0, default = 0
        Seconds to skip after a detected paek to look for new peaks.
    start_max : boolean, default = True
        Whether to start looking for a maximum or (False) for a minimum.

    Returns
    -------
    maxp : numpy.array
        Array containing indexes of the maxima
    minp : numpy.array
        Array containing indexes of the minima
    maxv : numpy.array
        Array containing values of the maxima
    minv : numpy.array
        Array containing values of the minima
    """
    
    _params_descriptors = {
        'delta': _Par(2, list, "Threshold for the detection of the peaks"),
        'refractory': _Par(0, float, "Seconds to skip after detection of a peak", 0, lambda x: x > 0),
        'start_max': _Par(0, bool, "Whether to start looking for a max.", True)
    }
    
    @classmethod
    def algorithm(cls, signal, params):
        refractory = params['refractory']
        if refractory == 0:
            refractory = 1
        else:
            refractory = refractory * signal.get_sampling_freq()
        look_for_max = params['start_max']
        delta = params['delta']
        
        if isinstance(delta, float) or isinstance(delta, int):
            deltas = _np.repeat(delta, len(signal))
        else:
            deltas = delta
        
        minp = []
        maxp = []

        minv = []
        maxv = []

        if len(signal) < 1:
            cls.warn("signal is too short (len < 1), returning empty.")
        elif len(deltas) != len(signal):
            cls.error("deltas vector's length differs from signal's one, returning empty.")
        else:
            mn_pos_candidate = mx_pos_candidate = 0
            mn_candidate = mx_candidate = signal[0]

            i_activation_min = 0
            i_activation_max = 0

            for i in range(1, len(signal)):
                sample = signal[i]
                d = deltas[i]

                if sample > mx_candidate:
                    mx_candidate = sample
                    mx_pos_candidate = i
                if sample < mn_candidate:
                    mn_candidate = sample
                    mn_pos_candidate = i

                if look_for_max:
                    if i >= i_activation_max and sample < mx_candidate - d:  # new max
                        maxp.append(mx_pos_candidate)
                        maxv.append(mx_candidate)
                        i_activation_max = i + refractory

                        mn_candidate = sample
                        mn_pos_candidate = i

                        look_for_max = False
                else:
                    if i >= i_activation_min and sample > mn_candidate + d:  # new min
                        minp.append(mn_pos_candidate)
                        minv.append(mn_candidate)
                        i_activation_min = i + refractory

                        mx_candidate = sample
                        mx_pos_candidate = i

                        look_for_max = True

        return _np.array(maxp), _np.array(minp), _np.array(maxv), _np.array(minv)


class PeakSelection(_Tool):
    """
    Identify the start and the end indexes of each peak in the signal, using derivatives.

    Parameters
    ----------
    maxs : array
        Array containing indexes (first column) and values (second column) of the maxima
    pre_max : float
        Duration (in seconds) of interval before the peak that is considered to find the start of the peak
    post_max : float
        Duration (in seconds) of interval after the peak that is considered to find the end of the peak
    
    Returns
    -------
    starts : array
        Array containing start indexes
    ends : array
        Array containing end indexes
    """

    _params_descriptors = {
        'maxs': _Par(2, list, 'Array containing indexes (first column) and values (second column) of the maxima'),
        'pre_max': _Par(2, float, 'Duration (in seconds) of interval before the peak that is considered to find the start of the peak', constraint= lambda x: x > 0),
        'post_max': _Par(2, float, 'Duration (in seconds) of interval after the peak that is considered to find the end of the peak', constraint= lambda x: x > 0)
    }
    
    @classmethod
    def algorithm(cls, signal, params):
        i_pre_max = int(params['pre_max'] * signal.get_sampling_freq())
        i_post_max = int(params['post_max'] * signal.get_sampling_freq())
        i_peaks = params['maxs']
        
        ZERO = 0.01 
        
        i_start = _np.empty(len(i_peaks), int)
        i_stop = _np.empty(len(i_peaks), int)

        if len(i_peaks)==0:
#            cls.warn(cls.__name__, ': No peaks given.')
            return i_start, i_stop
            
        signal_dt = _Diff()(signal)
        for i in xrange(len(i_peaks)):
            i_pk = int(i_peaks[i])

            if i_pk < i_pre_max or i_pk >= len(signal_dt) - i_post_max:
#                cls.log('Peak at start/end of signal, not accounting')

                i_start[i] = i_stop[i] = -1
            else:
                
                # find START
                i_st = i_pk - i_pre_max
                signal_dt_pre = signal_dt[i_st:i_pk]
                i_pre = len(signal_dt_pre) - 1

                while i_pre > 0 and (signal_dt_pre[i_pre] > 0 or abs(signal_dt_pre[i_pre]) <= ZERO):
                    i_pre -= 1

                i_start[i] = i_st + i_pre + 1

                # find STOP
                i_sp = i_pk + i_post_max
                signal_dt_post = signal_dt[i_pk: i_sp]
                i_post = 1

                while i_post < len(signal_dt_post)-1 and (signal_dt_post[i_post] < 0 or abs(signal_dt_post[i_post]) <= ZERO):
                    i_post += 1

                i_stop[i] = i_pk + i_post

        return i_start, i_stop


class SignalRange(_Tool):
    """
    Estimate the local range of the signal by sliding windowing

    Parameters
    ----------
    win_len : float, >0
        Length of the window  in seconds
    win_step : float, >0
        Shiftt to start the next window in seconds

    Optional:    
    smooth : boolean, default = True
        Whether to convolve the result with a gaussian window

    Returns
    -------
    deltas : numpy.array
        Local range of the signal
    """

    _params_descriptors = {
        'win_len': _Par(2, float, 'The length of the window (seconds)', constraint=lambda x: x > 0),
        'win_step': _Par(2, float, 'The increment to start the next window (seconds)', constraint=lambda x: x > 0),
        'smooth': _Par(0, bool, 'Whether to convolve the result with a gaussian window', True)
    }
    
    @classmethod
    def algorithm(cls, signal, params):
        win_len = params['win_len']
        win_step = params['win_step']
        smooth = params['smooth']

        fsamp = signal.get_sampling_freq()
        idx_len = int(win_len * fsamp)
        idx_step = int(win_step * fsamp)

        if len(signal) < idx_len:
            cls.warn("Input signal is shorter than the window length.")
            return _np.max(signal) - _np.min(signal)
        else:
            windows = _np.arange(0, len(signal) - idx_len + 1, idx_step)
            deltas = _np.zeros(len(signal))

            curr_delta = 0
            for start in windows:
                portion_curr = signal[start: start + idx_len]
                curr_delta = _np.max(portion_curr) - _np.min(portion_curr)
                deltas[start:start + idx_len] = curr_delta

            deltas[windows[-1] + idx_len:] = curr_delta

            deltas = _EvenlySignal(deltas, signal.get_sampling_freq())

            if smooth:
                deltas = _ConvFlt(irftype='gauss', win_len=win_len * 2, normalize=True)(deltas)

            return deltas.get_values()


class PSD(_Tool):
    """
    Estimate the power spectral density (PSD) of the signal.

    Parameters
    ----------
    method : str
        Method to estimate the PSD. Available methods: 'welch', 'fft', 'ar'
        
    Optional:
    nfft : int, >0, default=2048
        Number of samples of the PSD
    window : str, default = 'hamming'
        Type of window
    min_order : int, >0, default=18
        Minimum order of the model to be tested for psd_method='ar'
    max_order : int, >0, default=25
        Maximum order of the model to be tested for psd_method='ar'
    normalize : boolean, default = True
        Whether to normalize the PSD
    remove_mean : boolean, default = True
        Whether to remove the mean from the signal before estimating the PSD
    
    Returns
    -------
    freq : numpy.array
        Frequencies
    psd : numpy.array
        Power Spectrum Density
    """
    
    _method_list = ['welch', 'fft', 'ar']
    _window_list = ['hamming', 'blackman', 'hanning', 'bartlett', 'none']

    _params_descriptors = {
        'method': _Par(2, str, 'Method to estimate the PSD', constraint=lambda x: x in PSD._method_list), 
        'nfft': _Par(0, int, 'Number of samples in the PSD', 2048, lambda x: x > 0), 
        'window': _Par(0, str, 'Type of window to adapt the signal before estimation of the PSD', 'hamming', lambda x: x in PSD._window_list),
        'min_order': _Par(0, int, 'Minimum order of the model (for method="ar")', default=18, constraint=lambda x: x > 0, activation=lambda x, y: "method" in y and y["method"] == "ar"),
        'max_order': _Par(0, int, 'Maximum order of the model (for method="ar")', default=25, constraint=lambda x: x > 0, activation=lambda x, y: "method" in y and y["method"] == "ar"),
        'normalize': _Par(0, bool, 'Whether to normalize the PSD', True),
        'remove_mean': _Par(0, bool, 'Whether to remove the mean from the signal before estimation of the PSD', True)
    }
    
    # TODO (feature): consider point below:
    # A density spectrum considers the amplitudes per unit frequency.
    # Density spectra are used to compare spectra with different frequency resolution as the
    # magnitudes are not influenced by the resolution because it is per Hertz. The amplitude
    # spectra on the other hand depend on the chosen frequency resolution.

    @classmethod
    def algorithm(cls, signal, params):
        method = params['method']
        nfft = params['nfft'] if "nfft" in params else None
        window = params['window']
        normalize = params['normalize']
        remove_mean = params['remove_mean']
        
        if not isinstance(signal, _EvenlySignal):
            #TODO (feature) lomb scargle
            if len(signal) < 2: #zero or one sample: interpolation not allowed
                return _np.repeat(_np.nan, 2), _np.repeat(_np.nan, 2)
        
            interp_freq = params['interp_freq']
            signal = signal.to_evenly(kind='cubic')
            signal = signal.resample(interp_freq)

        fsamp = signal.get_sampling_freq()
        l = len(signal)
        if remove_mean:
            signal = signal - _np.mean(signal)

        if window == 'hamming':
            win = _np.hamming(l)
        elif window == 'blackman':
            win = _np.blackman(l)
        elif window == 'bartlett':
            win = _np.bartlett(l)
        elif window == 'hanning':
            win = _np.hanning(l)
        else:
            win = _np.ones(l)
            if window != 'none':
                cls.warn('Window type not understood, using none.')

        signal = signal * win
        if method == 'fft':
            spec_tmp = _np.abs(_np.fft.fft(signal, n=nfft)) ** 2  # FFT
            psd = spec_tmp[0:(_np.ceil(len(spec_tmp) / 2))]

        elif method == 'ar':
            min_order = params['min_order']
            max_order = params['max_order']
            
            # FIXME: min_order is None by default and not 18 as requested            
            #WORKAROUND for min_order = None by default
            if min_order is None:
                min_order = 18
            #END WORKAROUND
                
            orders = range(min_order, max_order + 1)
            aics = []
            for order in orders:
                try:
                    ar, p, k = _aryule(signal, order=order)
                    aics.append(_AIC(l, p, order))
                except AssertionError:
                    break
            best_order = orders[_np.argmin(aics)]

            ar, p, k = _aryule(signal, best_order)
            psd = _arma2psd(ar, NFFT=nfft)
            psd = psd[0: int(_np.ceil(len(psd) / 2))]

        elif method == 'welch':
            bands_w, psd = _welch(signal, fsamp, nfft=nfft)
        else:
            cls.warn('Method not understood, using welch.')
            bands_w, psd = _welch(signal, fsamp, nfft=nfft)

        freqs = _np.linspace(start=0, stop=fsamp / 2, num=len(psd))

        # NORMALIZE
        if normalize:
            psd /= 0.5 * fsamp * _np.sum(psd) / len(psd)
        return freqs, psd


class Maxima(_Tool):
    """
    Find all local maxima in the signal

    Parameters
    ----------
    method : str
        Method to detect the maxima. Available methods: 'complete' or 'windowing'. 
        'complete' finds all the local maxima, 'windowing' uses a runnning window to find the global maxima in each window.
    win_len : float, >0
        Length of window in seconds (method = 'windowing')
    win_step : float, >0
        Shift of the window to start the next window in seconds (method = 'windowing')
    
    Optional:
    refractory : float, >0, default = 0
        Seconds to skip after a detected maximum to look for new maxima, when method = 'complete'. 
    

    Returns
    -------
    idx_maxs : array
        Array containing indexes of the maxima
    val_maxs : array
        Array containing values of the maxima
    """
    
    _params_descriptors = {
        'method': _Par(2, str, 'Method to detect the maxima', 'complete', lambda x: x in ['complete', 'windowing']),        
        'refractory': _Par(0, float, 'Seconds to skip after detection of a maximum (method = "complete")', 0, lambda x: x > 0, lambda x, p: p['method'] == 'complete'),
        'win_len': _Par(2, float, "Length of window in seconds (method = 'windowing')", constraint=lambda x: x > 0, activation=lambda x, p: p['method'] == 'windowing'),
        'win_step': _Par(2, float, "Shift of the window to start the next window in seconds (method = 'windowing')", constraint=lambda x: x > 0, activation=lambda x, p: p['method'] == 'windowing'),
    }
    
    @classmethod
    def algorithm(cls, signal, params):
        method = params['method']
        
        if method == 'complete':
            refractory = params['refractory']
            if refractory == 0:
                refractory = 1
            else:
                refractory = refractory * signal.get_sampling_freq()
            idx_maxs = []
            prev = signal[0]
            k = 1
            while k < len(signal) - 1 - refractory:
                curr = signal[k]
                nxt = signal[k + 1]
                if (curr >= prev) and (curr >= nxt):
                    idx_maxs.append(k)
                    prev = signal[k + 1 + refractory]
                    k = k + 2 + refractory
                else:  # continue
                    prev = signal[k]
                    k += 1
            idx_maxs = _np.array(idx_maxs).astype(int)
            maxs = signal[idx_maxs]
            return idx_maxs, maxs

        elif method == 'windowing':
            fsamp = signal.get_sampling_freq()
            winlen = int(params['win_len'] * fsamp)
            winstep = int(params['win_step'] * fsamp)

            #TODO: check that winlen > 2
            #TODO: check that winstep >= 1

            idx_maxs = [_np.nan]
            maxs = [_np.nan]
            
            if winlen<len(signal):
                idx_start = _np.arange(0, len(signal)-winlen+1, winstep)
            else:
                idx_start = [0]

            for idx_st in idx_start:
                idx_sp = idx_st + winlen
                if idx_sp > len(signal):
                    idx_sp = len(signal)
                curr_win = signal[idx_st: idx_sp]
                curr_idx_max = _np.argmax(curr_win) + idx_st
                curr_max = _np.max(curr_win)

                # peak not already detected & peak not at the beginnig/end of the window:
                if curr_idx_max != idx_maxs[-1] and curr_idx_max != idx_st and curr_idx_max != idx_sp - 1:
                    idx_maxs.append(curr_idx_max)
                    maxs.append(curr_max)
            idx_maxs = idx_maxs[1:]
            maxs = maxs[1:]
            return _np.array(idx_maxs), _np.array(maxs)


class Minima(_Tool):
    """
    Find all local minima in the signal

    Parameters
    ----------
    method : str
        Method to detect the minima. Available methods: 'complete' or 'windowing'. 
        'complete' finds all the local minima, 'windowing' uses a runnning window to find the global minima in each window.
    win_len : float, >0
        Length of window in seconds (method = 'windowing')
    win_step : float, >0
        Shift of the window to start the next window in seconds (method = 'windowing')
    
    Optional:
    refractory : float, >0, default = 0
        Seconds to skip after a detected minimum to look for new minima, when method = 'complete'. 
    

    Returns
    -------
    idx_mins : array
        Array containing indexes of the minima
    val_mins : array
        Array containing values of the minima
    """

    @classmethod
    def algorithm(cls, signal, params):
        signal = signal.copy()
        signal *= -1
        idx_mins, mins = Maxima(**params)(signal)
        return idx_mins, -1*mins

    _params_descriptors = {
        'method': _Par(2, str, 'Method to detect the minima', constraint=lambda x: x in ['complete', 'windowing']),
        'refractory': _Par(0, float, 'Seconds to skip after detection of a minimum (method = "complete")', 0, lambda x: x >= 0, lambda x, p: p['method'] == 'complete'),
        'win_len': _Par(2, float, 'Size of window in seconds (method = "windowing")', constraint=lambda x: x > 0, activation=lambda x, p: p['method'] == 'windowing'),
        'win_step': _Par(2, float, 'Increment to start the next window in seconds (method = "windowing")', constraint=lambda x: x > 0, activation=lambda x, p: p['method'] == 'windowing'),
    }


class CreateTemplate(_Tool):
    """
    Create a template for matched filtering
    
    Parameters
    ----------
    ref_indexes : list of int
        Indexes of the signals to be used as reference point to generate the template
    idx_start : int, >0
        Index of the signal to start the segmentation of the portion used to generate the template
    idx_end : int, >0
        Index of the signal to end the segmentation of the portion used to generate the template
    smp_pre : int, >0
        Number of samples before the reference point to be used to generate the template
    smp_post : int, >0
        Number of samples after the reference point to be used to generate the template
    
    Returns
    -------
    template : numpy.array
        The template
    """

    _params_descriptors = {
        'ref_indexes': _Par(2, list, 'Indexes of the signals to be used as reference point to generate the template'),
        'idx_start': _Par(2, int, 'Index of the signal to start the segmentation of the portion used to generate the template', constraint=lambda x: x >= 0),
        'idx_stop': _Par(2, int, 'Index of the signal to end the segmentation of the portion used to generate the template', constraint=lambda x: x >= 0),
        'smp_pre': _Par(2, int, 'Number of samples before the reference point to be used to generate the template', constraint=lambda x: x > 0),
        'smp_post': _Par(2, int, 'Number of samples after the reference point to be used to generate the template', constraint=lambda x: x > 0)
    }
    
    @classmethod
    def algorithm(cls, signal, params):
        idx_start = params['idx_start']
        idx_end = params['idx_stop']
        smp_pre = params['smp_pre']
        smp_post = params['smp_post']
        ref_indexes = params['ref_indexes']
        sig = _np.array(signal[idx_start: idx_end])
        ref_indexes = _np.array(
            ref_indexes[_np.where((ref_indexes > idx_start) & (ref_indexes <= idx_end))[0]]) - idx_start
        total_samples = smp_pre + smp_post
        templates = _np.zeros(total_samples)

        for i in xrange(1, len(ref_indexes) - 1):
            idx_peak = ref_indexes[i]
            tmp = sig[idx_peak - smp_pre: idx_peak + smp_post]
            tmp = (tmp - _np.min(tmp)) / (_np.max(tmp) - _np.min(tmp))
            templates = _np.c_[templates, tmp]

        templates = templates[:, 1:]

        template = _np.mean(templates, axis=1)
        template = template - template[0]
        template = template / _np.sum(template)
        return template


class BootstrapEstimation(_Tool):
    """
    Perform a bootstrapped estimation of given statistical indicator
    
    Parameters
    ----------
    func : numpy function
        Function to use in the bootstrapping. Must accept data as input
        
    Optional:
    N : int, >0, default = 100
        Number of iterations
    k : float, (0,1), default = 0.5
        Portion of data to be used at each iteration
    
    Returns
    -------
    estim : float
        Bootstrapped estimate
    
    """
    from types import FunctionType as Func
    _params_descriptors = {
        'func': _Par(2, Func, 'Function (accepts as input a vector and returns a scalar).'),
        'N': _Par(0, int, 'Number of iterations', 100, lambda x: x > 0),
        'k': _Par(0, float, 'Portion of data to be used at each iteration', 0.5, lambda x: 0 < x < 1)
    }


    @classmethod
    def algorithm(cls, signal, params):
        signal = _np.asarray(signal)
        l = len(signal)
        func = params['func']
        niter = int(params['N'])
        k = params['k']

        estim = []
        for i in xrange(niter):
            ixs = _np.arange(l)
            ixs_p = _np.random.permutation(ixs)
            sampled_data = signal[ixs_p[:int(round(k * l))]]
            curr_est = func(sampled_data)
            estim.append(curr_est)
        return _np.mean(estim)


#TODO: remove this trivial function
class Durations(_Tool):
    """
    Compute durations of events starting from their start and stop indexes

    Parameters:
    -----------
    starts : list
        Start indexes along the data
    stops : list
        Stop indexes along the data

    Return:
    -------
    durations : list
        durations of the events
    """
    
    _params_descriptors = {
        "starts": _Par(2, list, "Start indexes along the data"),
        "stops": _Par(2, list, "Stop indexes along the data")
    }
    
    @classmethod
    def algorithm(cls, signal, params):
        starts = params["starts"]
        stops = params["stops"]

        fsamp = signal.get_sampling_freq()
        durations = []
        for I in xrange(len(starts)):
            if (stops[I] > 0) & (starts[I] >= 0):
                durations.append((stops[I] - starts[I]) / fsamp)
            else:
                durations.append(_np.nan)
        return durations


class Slopes(_Tool):
    """
    Compute rising slope of peaks

    Parameters:
    -----------
    starts : list
        Start of the peaks indexes
    peaks : list
        Peaks indexes

    Return:
    -------
    slopes : list
        Rising slopes the peaks
    """
    
    _params_descriptors = {
        "starts": _Par(2, list, "Start indexes along the data"),
        "peaks": _Par(2, list, "Peak indexes along the data")
    }
    
    @classmethod
    def algorithm(cls, data, params):
        starts = params["starts"]
        peaks = params["peaks"]

        fsamp = data.get_sampling_freq()
        slopes = []
        for I in xrange(len(starts)):
            if peaks[I] > 0 & starts[I]>=0:
                dy = data[peaks[I]] - data[starts[I]]
                dt = (peaks[I] - starts[I]) / fsamp
                slopes.append(dy / dt)
            else:
                slopes.append(_np.nan)
        return slopes


# IBI Tools
class BeatOutliers(_Tool):
    """
    Detects outliers in the IBI signal. 
    
    Parameters
    ----------
    
    Optional:
    
    cache : int, >0,  default = 3
        Nuber of IBI to be stored in the cache for adaptive computation of the interval of accepted values
    sensitivity : float, >0, default = 0.25
        Relative variation from the current IBI median value of the cache that is accepted
    ibi_median : float, >=0, default = 0
        IBI value use to initialize the cache. By default (ibi_median=0) it is computed as median of the input IBI
    
    Returns
    -------
    id_bad_ibi : numpy.array
        Identifiers of wrong beats
    
    Notes
    -----
    It only detects outliers. You should manually remove outliers using FixIBI
    
    """
    
    _params_descriptors = {
        'ibi_median': _Par(0, float, 'Ibi value used to initialize the cache. If 0 (default) the ibi_median is computed on the input signal', 0, lambda x: x >= 0),
        'cache': _Par(0, int, 'Number of IBI to be stored in the cache for adaptive computation of the interval of accepted values', 3, lambda x: x > 0),
        'sensitivity': _Par(0, float, 'Relative variation from the current median that is accepted', 0.25, lambda x: x > 0)
    }
    
    @classmethod
    def get_signal_type(cls):
        return ['IBI']

    @classmethod
    def algorithm(cls, signal, params):
        cache, sensitivity, ibi_median = params["cache"], params["sensitivity"], params["ibi_median"]
        
        if ibi_median == 0:
            ibi_expected = float(_np.median(signal))
        else:
            ibi_expected = float(ibi_median)
        
        id_bad_ibi = []
        ibi_cache = _np.repeat(ibi_expected, cache)
        counter_bad = 0

        # missings = []
        idx_ibi = signal.get_indices()
        ibi = signal.get_values()
        for i in xrange(1, len(idx_ibi)):
            curr_median = _np.median(ibi_cache)
            
            curr_ibi = ibi[i]

            if curr_ibi > curr_median * (1 + sensitivity):  # abnormal peak:
                id_bad_ibi.append(i)  # append ibi id to the list of bad ibi
                counter_bad += 1
            # missings.append([idx_ibi[i-1],idx_ibi[i]])

            elif curr_ibi < curr_median * (1 - sensitivity):  # abnormal peak:
                id_bad_ibi.append(i)  # append ibi id to the list of bad ibi
                counter_bad += 1
            else:
                ibi_cache = _np.r_[ibi_cache[1:], curr_ibi]
                counter_bad = 0
            if counter_bad == cache:  # ibi cache probably corrupted, reinitialize
                ibi_cache = _np.repeat(ibi_expected, cache)
                counter_bad = 0

        return id_bad_ibi


class FixIBI(_Tool):
    """
    Corrects the IBI series removing abnormal IBI
    
    Parameters
    ----------
    id_bad_ibi : array
        Identifiers of abnormal beats
   
    Returns
    -------
    ibi : Unevenly Signal
        Corrected IBI
            
    """
    
    _params_descriptors = {
        'id_bad_ibi': _Par(2, list, 'Identifiers of abnormal beats')
    }
    
    @classmethod
    def get_signal_type(cls):
        return ['IBI']

    @classmethod
    def algorithm(cls, signal, params):
        assert isinstance(signal, _UnevenlySignal), "IBI can only be represented by an UnevenlySignal, %s found." % type(
            signal)
        id_bad = params['id_bad_ibi']
        idx_ibi = signal.get_indices()
        ibi = signal.get_values()
        idx_ibi_nobad = _np.delete(idx_ibi, id_bad)
        ibi_nobad = _np.delete(ibi, id_bad)
        idx_ibi = idx_ibi_nobad.astype(int)
        ibi = ibi_nobad
        return _UnevenlySignal(ibi, signal.get_sampling_freq(), signal.get_signal_nature(), signal.get_start_time(), x_values = idx_ibi, x_type = 'indices')


class BeatOptimizer(_Tool):
    """
    Optimize detection of errors in IBI estimation.
    
    Parameters
    ----------
    
    Optional:
    
    B : float, >0, default = 0.25
        Ball radius in seconds to allow pairing between forward and backward beats
    cache : int, >0,  default = 3
        Nuber of IBI to be stored in the cache for adaptive computation of the interval of accepted values
    sensitivity : float, >0, default = 0.25
        Relative variation from the current IBI median value of the cache that is accepted
    ibi_median : float, >=0, default = 0
        IBI value use to initialize the cache. By default (ibi_median=0) it is computed as median of the input IBI
    
    Returns
        
    Returns
    -------
    ibi : UnevenlySignal
        Optimized IBI signal

    Notes
    -----
    See REF
    #TODO: insert REF      
    """
    # FIXME: (Andrea) Wrong first sample in the returned signal
    _params_descriptors = {
        'B': _Par(0, float, 'Ball radius in seconds to allow pairing between forward and backward beats', 0.25, lambda x: x > 0),
        'cache': _Par(0, int, 'Nuber of IBI to be stored in the cache for adaptive computation of the interval of accepted values', 5, lambda x: x > 0),
        'sensitivity': _Par(0, float, 'Relative variation from the current median that is accepted', 0.25, lambda x: x > 0),
        'ibi_median': _Par(0, int, 'Ibi value used to initialize the cache. If 0 (default) the ibi_median is computed on the input signal', 0, lambda x: x > 0)
    }

    @classmethod
    def get_signal_type(cls):
        return ['IBI']

    @classmethod
    def algorithm(cls, signal, params):
        b, cache, sensitivity, ibi_median = params["B"], params["cache"], params["sensitivity"], params["ibi_median"]

        idx_ibi = signal.get_indices()
        fsamp = signal.get_sampling_freq()

        if ibi_median == 0:
            ibi_expected = _np.median(_np.diff(idx_ibi))
        else:
            ibi_expected = ibi_median

        idx_st = idx_ibi[0]
        idx_ibi = idx_ibi - idx_st

        ###
        # RUN FORWARD:
        ibi_cache = _np.repeat(ibi_expected, cache)
        counter_bad = 0

        idx_1 = [idx_ibi[0]]
        ibi_1 = []

        prev_idx = idx_ibi[0]
        for i in xrange(1, len(idx_ibi)):
            curr_median = _np.median(ibi_cache)
            curr_idx = idx_ibi[i]
            curr_ibi = curr_idx - prev_idx

            if curr_ibi > curr_median * (1 + sensitivity):  # abnormal peak:
                prev_idx = curr_idx
                ibi_1.append(_np.nan)
                idx_1.append(curr_idx)
                counter_bad += 1
            elif curr_ibi < curr_median * (1 - sensitivity):  # abnormal peak:
                counter_bad += 1
            else:
                ibi_cache = _np.r_[ibi_cache[1:], curr_ibi]
                prev_idx = curr_idx
                ibi_1.append(curr_ibi)
                idx_1.append(curr_idx)

            if counter_bad == cache:  # ibi cache probably corrupted, reinitialize
                ibi_cache = _np.repeat(ibi_expected, cache)
                # action_message('Cache re-initialized - ' + str(curr_idx))  # , RuntimeWarning) # message
                counter_bad = 0

        ###
        # RUN BACKWARD:
        idx_ibi_rev = idx_ibi[-1] - idx_ibi
        idx_ibi_rev = idx_ibi_rev[::-1]

        ibi_cache = _np.repeat(ibi_expected, cache)
        counter_bad = 0

        idx_2 = [idx_ibi_rev[0]]
        ibi_2 = []

        prev_idx = idx_ibi_rev[0]
        for i in xrange(1, len(idx_ibi_rev)):
            curr_median = _np.median(ibi_cache)
            curr_idx = idx_ibi_rev[i]
            curr_ibi = curr_idx - prev_idx

            # print([curr_median*(1+sensitivity), curr_median*(1-sensitivity), curr_median])
            if curr_ibi > curr_median * (1 + sensitivity):  # abnormal peak:
                prev_idx = curr_idx
                ibi_2.append(_np.nan)
                idx_2.append(curr_idx)
                counter_bad += 1

            elif curr_ibi < curr_median * (1 - sensitivity):  # abnormal peak:
                counter_bad += 1
            else:
                ibi_cache = _np.r_[ibi_cache[1:], curr_ibi]
                prev_idx = curr_idx
                ibi_2.append(curr_ibi)
                idx_2.append(curr_idx)

            if counter_bad == cache:  # ibi cache probably corrupted, reinitialize
                ibi_cache = _np.repeat(ibi_expected, cache)
                # action_message('Cache re-initialized - ' + str(curr_idx))  # , RuntimeWarning) # OK Message
                counter_bad = 0

        idx_2 = -1 * (_np.array(idx_2) - idx_ibi_rev[-1])
        idx_2 = idx_2[::-1]
        ibi_2 = ibi_2[::-1]

        ###
        # add indexes of idx_ibi_2 which are not in idx_ibi_1 but close enough
        b = b * fsamp
        for i_2 in xrange(1, len(idx_2)):
            curr_idx_2 = idx_2[i_2]
            if not (curr_idx_2 in idx_1):
                i_1 = _np.where((idx_1 >= curr_idx_2 - b) & (idx_1 <= curr_idx_2 + b))[0]
                if not len(i_1) > 0:
                    idx_1 = _np.r_[idx_1, curr_idx_2]
        idx_1 = _np.sort(idx_1)

        ###
        # create pairs for each beat
        pairs = []
        for i_1 in xrange(1, len(idx_1)):
            curr_idx_1 = idx_1[i_1]
            if curr_idx_1 in idx_2:
                pairs.append([curr_idx_1, curr_idx_1])
            else:
                i_2 = _np.where((idx_2 >= curr_idx_1 - b) & (idx_2 <= curr_idx_1 + b))[0]
                if len(i_2) > 0:
                    i_2 = i_2[0]
                    pairs.append([curr_idx_1, idx_2[i_2]])
                else:
                    pairs.append([curr_idx_1, curr_idx_1])
        pairs = _np.array(pairs)

        ########################################
        # define zones where there are different values
        diff_idxs = pairs[:, 0] - pairs[:, 1]
        diff_idxs[diff_idxs != 0] = 1
        diff_idxs = _np.diff(diff_idxs)

        starts = _np.where(diff_idxs > 0)[0]
        stops = _np.where(diff_idxs < 0)[0]
        
        if len(starts)==0: # no differences
            return signal
        
        if len(stops)==0:
            stops = _np.array([starts[-1] + 1])
            
        if starts[0] >= stops[0]:
            stops = stops[1:]

        stops += 1

        if len(starts) > len(stops):
            stops = _np.r_[stops, starts[-1] + 1]

        # split long sequences
        new_starts = _np.copy(starts)
        new_stops = _np.copy(stops)

        add_index = 0
        lens = stops - starts
        for i in xrange(len(starts)):
            l = lens[i]
            if l > 10:
                curr_st = starts[i]
                curr_sp = stops[i]
                new_st = _np.arange(curr_st, curr_sp, 4)
                new_sp = new_st + 4
                new_sp[-1] = curr_sp
                new_starts = _np.delete(new_starts, i + add_index)
                new_stops = _np.delete(new_stops, i + add_index)
                new_starts = _np.insert(new_starts, i + add_index, new_st)
                new_stops = _np.insert(new_stops, i + add_index, new_sp)
                add_index = add_index + len(new_st) - 1

        starts = new_starts
        stops = new_stops

        ########################################
        # find best combination
        idx_out = _np.copy(pairs[:, 0])
        for i in xrange(len(starts)):
            i_st = starts[i]
            i_sp = stops[i]

            if i_sp > len(idx_out) - 1:
                i_sp = len(idx_out) - 1

            curr_portion = _np.copy(pairs[i_st - 1: i_sp + 1, :])

            best_portion = None
            best_error = _np.Inf

            combinations = list(_itertools.product([0, 1], repeat=i_sp - i_st - 1))
            for comb in combinations:
                cand_portion = _np.copy(curr_portion[:, 0])
                for k in xrange(len(comb)):
                    bit = comb[k]
                    cand_portion[k + 2] = curr_portion[k + 2, bit]
                cand_error = sum(abs(_np.diff(_np.diff(cand_portion))))
                if cand_error < best_error:
                    best_portion = cand_portion
                    best_error = cand_error
            idx_out[i_st - 1: i_sp + 1] = best_portion

        ###
        # finalize arrays
        idx_out = _np.array(idx_out) + idx_st
        ibi_out = _np.diff(idx_out)
        ibi_out = _np.r_[ibi_out[0], ibi_out]

        return _UnevenlySignal(ibi_out/signal.get_sampling_freq(), signal.get_sampling_freq(), "IBI", signal.get_start_time(), x_values = idx_out, x_type = 'indices')


# EDA Tools
class OptimizeBateman(_Tool):
    """
    Optimize the Bateman parameters T1 and T2.
    
    Parameters
    ----------
    delta : float
        Minimum amplitude of the peaks in the driver
        
    Optional:
    
    opt_method : str
        Method to perform the search of optimal parameters.
        Available methods:
        - 'asa' Adaptive Simulated Annealing. Uses the algorithm proposed in REF
        #TODO: insert ref
        - 'grid' Grid search
    complete : boolean, default = True
        Whether to perform minimization after detecting the optimal parameters
    par_ranges : list, default = [0.1, 0.99, 1, 10]
        [min_T1, max_T1, min_T2, max_T2] boundaries for the Bateman parameters
    maxiter : int (Default = 99999)
        Maximum number of iterations ('asa' method)
    n_step : int
        Number of steps in the grid search
    weight : str
        How the errors should be weighted before computing the loss function. ['exp', 'lin', 'none']
    min_pars : dict
        Additional parameters to pass to the minimization function (when complete = True)

    Returns
    -------
    x0 : list
        The resulting optimal parameters
    x0_min : list
        If complete = True, parameters resulting from the minimization
    """

    #TODO: fix parameters
    _params_descriptors = {
        'delta': _Par(2, float, 'Minimum amplitude of the peaks in the driver', default = None, constraint=lambda x: x > 0),
        'loss_func': _Par(0, str, 'Loss function to be used', default='bizzego', constraint=lambda x: x in ['bizzego', 'benedek', 'all']),
        'opt_method': _Par(0, str, 'Method to perform the search of optimal parameters.', default='asa', constraint=lambda x: x in ['grid', 'bsh']),
        'complete': _Par(0, bool, 'Whether to perform a minimization after detecting the optimal parameters', default=True), 
        'par_ranges': _Par(0, list, '[min_T1, max_T1, min_T2, max_T2] boundaries for the Bateman parameters', default=[0.1, 0.99, 1.5, 5], constraint = lambda x: len(x) == 4),
        'maxiter': _Par(0, int, 'Maximum number of iterations ("asa" method).', default=99999, constraint=lambda x: x > 0, activation=lambda x, p: p['opt_method'] == 'asa'),
        'n_step': _Par(0, int, 'Number of steps in the grid search', default=10, constraint=lambda x: x > 0, activation=lambda x, p: p['opt_method'] == 'grid'),
        'weight' : _Par(0, str, 'How the errors should be weighted before computing the loss function', default='none', constraint=lambda x: x in ['exp', 'lin', 'none']),
        'fmin_params': _Par(0, dict, 'Additional parameters to pass to the minimization function (when complete = True)', default={}, activation=lambda x, p: p['complete'])
    }
    
    @classmethod
    def algorithm(cls, signal, params):
        delta = params['delta']
        opt_method = params['opt_method']
        complete = params['complete']
        par_ranges = params['par_ranges']
        maxiter = params['maxiter']
        n_step = params['n_step']
        
        weight = params['weight']
        min_pars = params['fmin_params']
        
        if params['loss_func'] == 'benedek':
            loss_function = OptimizeBateman._loss_benedek
        elif params['loss_func'] == 'all':
            loss_function = OptimizeBateman._loss_function_all
        else:
            loss_function = OptimizeBateman._loss_function

        min_T1 = float(par_ranges[0])
        max_T1 = float(par_ranges[1])

        min_T2 = float(par_ranges[2])
        max_T2 = float(par_ranges[3])
        
        if opt_method == 'grid':
            step_T1 = (max_T1 - min_T1) / n_step
            step_T2 = (max_T2 - min_T2) / n_step
            rranges = (slice(min_T1, max_T1 + step_T1, step_T1), slice(min_T2, max_T2 + step_T2, step_T2))
            x0, loss, grid, loss_grid = _opt.brute(loss_function, rranges, args=(signal, delta, min_T1, max_T1, min_T2, max_T2, weight),
                                                      finish=None, full_output=True)
            exit_code = -1
        elif opt_method == 'bsh':
            x_opt = _opt.basinhopping(loss_function, [0.75, 2.], niter=maxiter, minimizer_kwargs = {"bounds":((par_ranges[0],par_ranges[1]),(par_ranges[2],par_ranges[3])), "args":(signal, delta, min_T1, max_T1, min_T2, max_T2, weight), "tol":0.001}, disp=False, niter_success=10)
            x0 = x_opt.x
            loss = float(x_opt.fun)
            
            if x_opt.minimization_failures == maxiter:
                exit_code = 1
            else:
                exit_code = 0
        else:
            cls.error("opt_method not understood")
            return None
        
        if complete:
            x0_min, loss_min, niter, nfuncalls, warnflag = _opt.fmin(loss_function, x0, args=(signal, delta, min_T1, max_T1, min_T2, max_T2, weight),
                                                                     full_output=True,
                                                                         **min_pars)
            return x0, x0_min, loss, loss_min, exit_code, warnflag
        else:
            return x0, loss, exit_code

    @staticmethod
    def _loss_function(par_bat, signal, delta, min_T1, max_T1, min_T2, max_T2, weight):
        """
        Computes the loss for optimization of Bateman parameters.

        Parameters
        ----------
        par_bat : list
            Bateman parameters to be optimized
        signal : array
            The EDA signal
        delta : float
            Minimum amplitude of the peaks in the driver
        min_T1 : float
            Lower bound for T1
        max_T1 : float
            Upper bound for T1
        min_T2 : float
            Lower bound for T2
        max_T2 : float
            Upper bound for T2
        weight : str, default = 'none'
            How to weight the errors in the driver function before computing the loss: 'exp', 'lin' o 'none'
       
        Returns
        -------
        loss : float
            The computed loss
        """
        from ..estimators.Estimators import DriverEstim as _DriverEstim
        
        if _np.isnan(par_bat[0]) | _np.isnan(par_bat[1]):
            return _np.Inf
        
        WLEN = 10
        # check if pars hit boudaries
#        if par_bat[0] < min_T1 or par_bat[0] > max_T1 or par_bat[1] < min_T2 or par_bat[1] > max_T2 or par_bat[0] >= par_bat[1]:
#            return 10000 

        fsamp = signal.get_sampling_freq()
        driver = _DriverEstim(T1=par_bat[0], T2=par_bat[1])(signal)
        maxp, minp, ignored, ignored = PeakDetection(delta=delta, start_max=True)(driver)

        if len(maxp) == 0:
            OptimizeBateman.warn('Unable to find peaks in driver signal for computation of Energy. Returning Inf')
            return _np.inf
        else:
            # STAGE 1: select maxs distant from the others
            diff_maxs = _np.diff(_np.r_[maxp, len(driver) - 1])
            th_diff = WLEN * fsamp

            # TODO (feature): select th such as to have enough maxs, e.g. diff_maxs_tentative = np.median(diff_maxs)
            idx_selected_maxs = _np.where(diff_maxs > th_diff)[0]
            selected_maxs = maxp[idx_selected_maxs]

            if len(selected_maxs) != 0:
                energy = 0
                for idx_max in selected_maxs:
                    #extract WLEN seconds after the peak
                    driver_portion = driver[idx_max:idx_max + WLEN * fsamp]

                    #extract final 5 seconds
                    half = len(driver_portion) - 5 * fsamp
                    
                    #estimate slope to detrend driver
                    y = driver_portion[half:]
                    diff_y = _np.diff(y)
                    th_75 = _np.percentile(diff_y, 75)
                    th_25 = _np.percentile(diff_y, 25)

                    idx_sel_diff_y = _np.where((diff_y > th_25) & (diff_y < th_75))[0]
                    diff_y_sel = diff_y[idx_sel_diff_y]

                    mean_s = BootstrapEstimation(func=_np.mean, N=10, k=0.5)(diff_y_sel)

                    mean_y = BootstrapEstimation(func=_np.median, N=10, k=0.5)(y)

                    b_mean_s = mean_y - mean_s * (half + (len(driver_portion) - half) / 2)

                    line_mean_s = mean_s * _np.arange(len(driver_portion)) + b_mean_s

                    driver_detrended = driver_portion - line_mean_s

                    #normalize detrended driver
                    driver_detrended /= _np.max(driver_detrended)
                    
                    #weighting
                    if weight == 'exp':
                        idxs = _np.arange(len(driver_detrended))
                        exp_weigths = _np.exp(-idxs / (par_bat[0]*fsamp))
                        driver_detrended = (driver_detrended - _np.mean(driver_detrended)) * exp_weigths + _np.mean(driver_detrended)
                    elif weight == 'lin':
                        lin_weigths = _np.arange(1, 0, -1/len(driver_detrended))
                        driver_detrended = (driver_detrended - _np.mean(driver_detrended)) * lin_weigths + _np.mean(driver_detrended)

                    energy_curr = (1 / fsamp) * _np.sum(driver_detrended[1:] ** 2) / (len(driver_detrended) - 1)
#                    energy_curr = (1 / fsamp) * _np.sum(driver_detrended[fsamp:] ** 2) / (len(driver_detrended) - fsamp)

                    energy += energy_curr

            else:
                OptimizeBateman.warn('Peaks found but too near. Returning Inf')
                return _np.inf
            
            #normalize to the number of peaks
            energy /= len(selected_maxs)
            OptimizeBateman.log('BIZZEGO. Current parameters: ' + str(par_bat[0]) + ' - ' + str(par_bat[1]) + ' Loss: ' + str(energy))
            
            return energy

    @staticmethod
    def _loss_function_all(par_bat, signal, delta, min_T1, max_T1, min_T2, max_T2, weight):
        """
        Computes the loss for optimization of Bateman parameters.

        Parameters
        ----------
        par_bat : list
            Bateman parameters to be optimized
        signal : array
            The EDA signal
        delta : float
            Minimum amplitude of the peaks in the driver
        min_T1 : float
            Lower bound for T1
        max_T1 : float
            Upper bound for T1
        min_T2 : float
            Lower bound for T2
        max_T2 : float
            Upper bound for T2
        weight : str, default = 'none'
            How to weight the errors in the driver function before computing the loss: 'exp', 'lin' o 'none'
       
        Returns
        -------
        loss : float
            The computed loss
        """

        from ..estimators.Estimators import DriverEstim as _DriverEstim
        # check if pars hit boudaries
        #if par_bat[0] < min_T1 or par_bat[0] > max_T1 or par_bat[1] < min_T2 or par_bat[1] > max_T2 or par_bat[0] >= par_bat[1]: 
#            return 10000 

        WLEN = 10
        if _np.isnan(par_bat[0]) | _np.isnan(par_bat[1]):
            return _np.Inf
            
        fsamp = signal.get_sampling_freq()

        driver = _DriverEstim(T1=par_bat[0], T2=par_bat[1])(signal)
        maxp, minp, ignored, ignored = PeakDetection(delta=delta, start_max=True)(driver[:-15*fsamp])

        if len(maxp) == 0:
            OptimizeBateman.warn('Unable to find peaks in driver signal for computation of Energy. Returning Inf')
            return _np.inf
        else:
            energy = 0
            peak_counts = 0
        
            for idx_max in maxp:
                #extract WLEN seconds after the peak
                driver_portion = driver[idx_max:idx_max + WLEN * fsamp]
                
#                lin_weigths = _np.arange(1, 0, -1/len(driver_portion))
#                driver_portion = (driver_portion - _np.mean(driver_portion)) * lin_weigths + _np.mean(driver_portion)
#
#                idxs = _np.arange(len(driver_portion))
#                exp_weigths = _np.exp(-idxs / (par_bat[0]*fsamp))
#                driver_portion = (driver_portion - _np.mean(driver_portion)) * exp_weigths + _np.mean(driver_portion)
                
                ##estimate trend
                #extract final 5 seconds
#                half = len(driver_portion) - 10 * fsamp
                
                #estimate slope from final part
                
                maxp, minp, ignored, minv = PeakDetection(delta=delta/10, start_max=True)(driver_portion)
                if len(minp)>4:
                    y = driver_portion[minp]
                    t = minp
                    
                    diff_y = (y[1:] - y[:-1])/(t[1:] - t[:-1])

                    th_75 = _np.percentile(diff_y, 75)
                    th_25 = _np.percentile(diff_y, 25)
    
                    idx_sel_diff_y = _np.where((diff_y > th_25) & (diff_y < th_75))[0]
                    diff_y_sel = diff_y[idx_sel_diff_y]
    
                    mean_s = BootstrapEstimation(func=_np.mean, N=10, k=0.5)(diff_y_sel)
    
                    mean_y = BootstrapEstimation(func=_np.median, N=10, k=0.5)(y)
    
                    b_mean_s = mean_y - mean_s * len(driver_portion)/2
    
                    line_mean_s = mean_s * _np.arange(len(driver_portion)) + b_mean_s
    
                    driver_detrended = driver_portion - line_mean_s
    
                    #normalize detrended driver
                    driver_detrended /= float(_np.max(driver_detrended))
                    
                    energy_curr = (1 / fsamp) * _np.sum(driver_detrended[1:] ** 2) / (len(driver_detrended) - 1)
                    
                    energy += energy_curr
                    peak_counts +=1

            if peak_counts == 0:
                return _np.Inf
            energy /= peak_counts
            OptimizeBateman.log('ALL. Current parameters: ' + str(par_bat[0]) + ' - ' + str(par_bat[1]) + ' Loss: ' + str(energy))
            
            if _np.isnan(energy):
                return _np.Inf

            return energy

    @staticmethod
    def _loss_benedek(par_bat, signal, delta, min_T1, max_T1, min_T2, max_T2, weight):
        """
        Computes the loss for optimization of Bateman parameters according to Benedek2010 (REF).
        #TODO: insert reference

        Parameters
        ----------
        par_bat : list
            Bateman parameters to be optimized
        signal : array
            The EDA signal
        delta : float
            Minimum amplitude of the peaks in the driver
        min_T1 : float
            Lower bound for T1
        max_T1 : float
            Upper bound for T1
        min_T2 : float
            Lower bound for T2
        max_T2 : float
            Upper bound for T2
        alpha : float, default = 6
            Coefficient to weight the 'neg' component
       
        Returns
        -------
        loss : float
            The computed loss
        """
        
        from ..estimators.Estimators import DriverEstim as _DriverEstim

        if _np.isnan(par_bat[0]) | _np.isnan(par_bat[1]):
            return _np.Inf
            
        def phasic_estim_benedek(driver, delta):
            #find peaks in the driver
            fsamp = driver.get_sampling_freq()
            i_peaks, idx_min, val_max, val_min = PeakDetection(delta=delta, refractory=1, start_max=True)(driver)
            
            i_pre_max = 10 * fsamp
            i_post_max = 10 * fsamp
                
            
            i_start = _np.empty(len(i_peaks), int)
            i_stop = _np.empty(len(i_peaks), int)
        
            if len(i_peaks)==0:
                print('No peaks found.')
                return driver
                
            for i in xrange(len(i_peaks)):
                i_pk = int(i_peaks[i])
        
                # find START
                i_st = i_pk - i_pre_max
                if i_st < 0:
                    i_st=0
        
                driver_pre = driver[i_st:i_pk]
                i_pre = len(driver_pre) -2
        
                while i_pre > 0 and (driver_pre[i_pre] >= driver_pre[i_pre-1]):
                    i_pre -= 1
        
                i_start[i] = i_st + i_pre + 1
                
        
                # find STOP
                i_sp = i_pk + i_post_max
                if i_sp >= len(driver):
                    i_sp = len(driver) - 1
                    
                driver_post = driver[i_pk: i_sp]
                i_post = 1
        
                while i_post < len(driver_post)-2 and (driver_post[i_post] >= driver_post[i_post+1]):
                    i_post += 1
        
                i_stop[i] = i_pk + i_post
            
            idxs_peak = _np.array([])
            
            for i_st, i_sp in zip(i_start, i_stop):
                idxs_peak = _np.r_[idxs_peak, _np.arange(i_st, i_sp)]
            
            idxs_peak = idxs_peak.astype(int)
            
            #generate the grid for the interpolation
            idx_grid_candidate = _np.arange(0, len(driver) - 1, 10 * fsamp)
        
            idx_grid = []
            for i in idx_grid_candidate:
                if i not in idxs_peak:
                    idx_grid.append(i)
            
            if len(idx_grid)==0:
                idx_grid.append(0)
            
            if idx_grid[0] != 0:
                idx_grid = _np.r_[0, idx_grid]
            if idx_grid[-1] != len(driver) - 1:
                idx_grid = _np.r_[idx_grid, len(driver) - 1]
        
            driver_grid = _UnevenlySignal(driver[idx_grid], fsamp, "dEDA", driver.get_start_time(), x_values = idx_grid, x_type = 'indices')
            if len(idx_grid)>=4:
                tonic = driver_grid.to_evenly(kind='cubic')
            else:
                tonic = driver_grid.to_evenly(kind='linear')
            phasic = driver - tonic
            return phasic
        
        ALPHA = 6
        driver = _DriverEstim(T1=par_bat[0], T2=par_bat[1])(signal)
        
        phasic = phasic_estim_benedek(driver, delta)
        print(len(phasic))
        
        TH = float(_np.max(phasic)*0.05)
        
        peaks = _np.zeros(len(phasic))
        peaks[phasic>TH] = 1
        
        # compute indistinctness
        dpeaks_dt = _np.diff(peaks)
        
        starts = _np.where(dpeaks_dt==1)[0]
        ends = _np.where(dpeaks_dt==-1)[0]
        
        if (len(starts) == 0) | (len(ends) == 0):
            indist = _np.Inf
        else:
            if ends[0] < starts[0]: #phasic starts with a peak
                starts = _np.r_[0, starts]
            
            if ends[-1] < starts[-1]: #phasic ends with a peak
                ends = _np.r_[ends, len(peaks)-1]
                
            #at this point is should be len(starts)==len(ends)
            indist = _np.sum((ends - starts)**2)/phasic.get_duration()
        
        # compute negativeness
        negatives = phasic[phasic<0]
        neg = _np.sqrt(_np.mean(negatives**2))
    
        # compute loss
        LOSS = indist + ALPHA * neg
        OptimizeBateman.log('BENEDEK. Current parameters: ' + str(par_bat[0]) + ' - ' + str(par_bat[1]) + ' Loss: ' + str(LOSS))
        return LOSS

#TODO: remove Histogram class
class Histogram(_Tool):
    """
    Compute the histogram of a set of data.
    
    Parametes:
    ----------
    
    Optional:
    
    histogram_bins : list
        'Number of bins (int) or bin edges, including the rightmost edge (list-like).'
    
    Returns:
    histogram
    """
        
    _params_descriptors = {
        'histogram_bins': _Par(1, list,
                               'Number of bins (int) or bin edges, including the rightmost edge (list-like).', 100,
                               lambda x: type(x) is not int or x > 0)
    }
    
    @classmethod
    def algorithm(cls, signal, params):
        return _np.histogram(signal, params['histogram_bins'])