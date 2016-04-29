# coding=utf-8
from __future__ import division
import numpy as _np
from ..BaseEstimator import Estimator as _Estimator
from ..Signal import UnevenlySignal as _UnevenlySignal, EvenlySignal as _EvenlySignal
from ..Utility import PhUI as _PhUI
from ..filters.Filters import IIRFilter as _IIRFilter, Diff as _Diff, DeConvolutionalFilter as _DeConvolutionalFilter, \
    ConvolutionalFilter as _ConvolutionalFilter
from ..tools.Tools import SignalRange as _SignalRange, PeakDetection as _PeakDetection, Minima as _Minima, \
    PeakSelection as _PeakSelection
from ..Parameters import Parameter as _Par

__author__ = 'AleB'


# IBI ESTIMATION
class BeatFromBP(_Estimator):
    """
    Identify the beats in a Blood Pulse (BP) signal and compute the IBIs.
    Optimized to identify the percussion peak.

    Limitations: works only on 'BP' type signal.

    Based on two stages:
    1) Identification of candidate beats
    2) Identification of the peak for each beat, using the derivative of the signal

    Parameters
    ----------
    bpm_max : int (>0)
        Maximal expected heart rate (in beats per minute)

    Returns
    -------
    ibi : UnevenlySignal
        Inter beat interval values at percussion peaks

    Notes
    -----
        ...
    """

    @classmethod
    def get_signal_type(cls):
        return ['BVP', 'BP']

    @classmethod
    def algorithm(cls, signal, params):  # FIX others TODO Andrea: ?
        bpm_max = params["bpm_max"]
        # method = params["method"]
        # sigma = params["sigma"]

        fmax = bpm_max / 60

        fsamp = signal.get_sampling_freq()

        refractory = int(fsamp / fmax)

        # STAGE 1 - EXTRACT BEAT POSITION SIGNAL
        signal_f = _IIRFilter(fp=1.2 * fmax, fs=3 * fmax)(signal)

        deltas = 0.5 * _SignalRange(win_len=3 / fmax, win_step=1 / fmax)(signal)

        # detection of candidate peaks
        maxp, minp = _PeakDetection(deltas=deltas, refractory=refractory, start_max=True)(signal_f)  # Tools
        idx_d = maxp[:, 0]

        if idx_d[0] == 0:
            idx_d = idx_d[1:]

        # STAGE 3 - IDENTIFY PEAKS using the signal derivative
        dxdt = _Diff()(signal)
        true_peaks = []

        WIN = 0.25 * fsamp
        for idx_beat in idx_d:
            start_ = idx_beat - WIN
            if start_ < 0:
                start_ = 0

            # select portion of derivative where to search
            obs = dxdt[start_:idx_beat]
            peak_obs = _np.argmax(obs)
            true_obs = dxdt[start_ + peak_obs: idx_beat]

            # find the 'first minimum' (zero) the derivative (peak)
            mins = _Minima(win_len=0.05, win_step=0.0025, method='windowing')(abs(true_obs))
            idx_mins = mins[:, 0]
            if len(idx_mins) >= 2:
                peak = idx_mins[1]
                true_peaks.append(start_ + peak_obs + peak)
            else:
                _PhUI.w('Peak not found; idx_beat: ' + str(idx_beat))
                pass

        # STAGE 4 - FINALIZE computing IBI and fixing indexes
        ibi_values = _Diff()(true_peaks) / fsamp
        ibi_values = _np.r_[ibi_values[0], ibi_values]
        idx_ibi = _np.array(true_peaks)

        ibi = _UnevenlySignal(ibi_values, idx_ibi, fsamp, 'IBI', signal.get_start_time(), signal.get_metadata())
        return ibi

    _params_descriptors = {
        'bpm_max': _Par(1, int, 'Maximal expected heart rate (in beats per minute)', 180, lambda x: 1 < x <= 400)
    }

    @staticmethod
    def _generate_gaussian_derivative(M, S):
        # TODO (Andrea): _gaussian not found
        g = _gaussian(M, S)
        gaussian_derivative_model = _np.diff(g)

        return gaussian_derivative_model


class BeatFromECG(_Estimator):
    """
    Identify the beats in an ECG signal and compute the IBIs.

    Parameters
    ----------
    bpm_max : int (>0)
        Maximal expected heart rate (in beats per minute)
        If not given the peak detection will run without refractory period.
    delta : float (default=0)
        Threshold for the peak detection.
        If not given or delta=0 it will be computed from the signal.

    Returns
    -------
    idx_ibi : nparray
        Indexes of the input signal corresponding to R peaks
    ibi : nparray
        Inter beat interval values at R peaks

    Notes
    -----
    See paper ...
    """

    @classmethod
    def get_signal_type(cls):
        return ['ECG']

    @classmethod
    def algorithm(cls, signal, params):
        bpm_max, delta = params["bpm_max"], params["delta"]

        fmax = bpm_max / 60

        if delta == 0:
            delta = 0.7 * _SignalRange(win_len=2/fmax, win_step=0.5/fmax)(signal)
        else:
            delta = _np.repeat(delta, len(signal))

        fsamp = signal.get_sampling_freq()

        refractory = int(fsamp / fmax)

        maxp, minp = _PeakDetection(delta=delta, refractory=refractory, start_max=True)(signal)

        idx_d = maxp[:, 0]

        if idx_d[0] == 0:
            idx_d = idx_d[1:]

        ibi_values = _Diff()(idx_d) / fsamp
        ibi_values = _np.r_[ibi_values[0], ibi_values]
        idx_ibi = _np.array(idx_d)

        ibi = _UnevenlySignal(ibi_values, idx_ibi, fsamp, 'IBI', signal.get_start_time(), signal.get_metadata())
        return ibi

    _params_descriptors = {
        'bpm_max': _Par(1, int, 180, 1, 'Maximal expected heart rate (in beats per minute)', lambda x: x > 0),
        'delta': _Par(1, float, 'Threshold for the peak detection. If delta = 0 (default) the signal range'
                                ' is automatically computed and used',
                      0, lambda x: x > 0)
    }
    
    #FIXME: ibi_estimator = est_new.BeatFromECG(bpm_max = 180) # ERRORE



# PHASIC ESTIMATION
class DriverEstim(_Estimator):
    """
    Estimates the driver of an EDA signal according to Benedek 2010

    The estimation uses a deconvolution using a bateman function as Impulsive Response Function.
    Based on the bateman function:

    :math:`b = e^{-t/t1} - e^{-t/t2}`

    Parameters
    ----------
    par_bat: list (t1, t2)
        Parameters of the bateman function

    Returns
    -------
    driver : EvenlySignal
        The driver function

    Notes
    -----
    ...
    """

    @classmethod
    def get_signal_type(cls):
        return ['EDA']

    @classmethod
    def algorithm(cls, signal, params):
        t1 = params['t1']
        t2 = params['t2']

        par_bat = [t1, t2]

        fsamp = signal.fsamp

        bateman = DriverEstim._gen_bateman(fsamp, par_bat)

        idx_max_bat = _np.argmax(bateman)

        # Prepare the input signal to avoid starting/ending peaks in the driver
        bateman_first_half = bateman[0:idx_max_bat + 1]
        bateman_first_half = signal[0] * (bateman_first_half - _np.min(bateman_first_half)) / (
            _np.max(bateman_first_half) - _np.min(bateman_first_half))

        bateman_second_half = bateman[idx_max_bat:]
        bateman_second_half = signal[-1] * (bateman_second_half - _np.min(bateman_second_half)) / (
            _np.max(bateman_second_half) - _np.min(bateman_second_half))

        signal_in = _np.r_[bateman_first_half, signal, bateman_second_half]

        # deconvolution
        driver = _DeConvolutionalFilter(irf=bateman)(signal_in)
        driver = driver[idx_max_bat + 1: idx_max_bat + len(signal)]

        # gaussian smoothing
        driver = _ConvolutionalFilter(irftype='gauss', win_len=0.2 * 8)

        driver = _EvenlySignal(driver, fsamp, "dEDA", signal.get_start_time(), signal.get_metadata())
        return driver

    _params_descriptors = {
        'T1': _Par(1, float, 'T1 parameter for the Bateman function', 0.75, lambda x: x > 0),
        'T2': _Par(1, float, 'T2 parameter for the Bateman function', 2, lambda x: x > 0)
    }

    @staticmethod
    def _gen_bateman(fsamp, par_bat):
        """
        Generates the bateman function:

        :math:`b = e^{-t/T1} - e^{-t/T2}`

        Parameters
        ----------
        fsamp : float
            The sampling frequency
        par_bat: list (T1, T2)
            Parameters of the bateman function

        Returns
        -------
        bateman : nparray
            The bateman function
        """

        idx_T1 = par_bat[0] * fsamp
        idx_T2 = par_bat[1] * fsamp
        len_bat = idx_T2 * 10
        idx_bat = _np.arange(len_bat)
        bateman = _np.exp(-idx_bat / idx_T2) - _np.exp(-idx_bat / idx_T1)

        # normalize
        bateman = fsamp * bateman / _np.sum(bateman)
        return bateman


class PhasicEstim(_Estimator):
    """
    Estimates the phasic and tonic components of a driver.

    It uses a detection algorithm based on the derivative of the driver.

    Parameters
    ----------
    delta : float
        Minimum amplitude of the peaks in the driver
    grid_size : float
        Sampling size of the interpolation grid
    pre_max : float
        Duration (in seconds) of interval before the peak that is considered to find the start of the peak
    post_max : float
        Duration (in seconds) of interval after the peak that is considered to find the end of the peak

    Returns
    -------
    tonic : EvenlySignal
        The tonic component
    phasic : EvenlySignal
        The phasic component
    driver_no_peak : EvenlySignal
        The "de-peaked" driver signal used to generate the interpolation grid
    """

    @classmethod
    def get_signal_type(cls):
        return ['dEDA']

    @classmethod
    def algorithm(cls, signal, params):
        delta = params["delta"]
        grid_size = params["grid_size"]
        pre_max = params['pre_max']
        post_max = params['post_max']

        fsamp = signal.fsamp
        max_driv, tmp_ = _PeakDetection(delta=delta, refractory=1, start_max=True)(signal)

        idx_pre, idx_post = _PeakSelection(maxs=max_driv, pre_max=pre_max, post_max=post_max)

        # Linear interpolation to substitute the peaks
        driver_no_peak = _np.copy(signal)
        for I in range(len(idx_pre)):
            i_st = idx_pre[I]
            i_sp = idx_post[I]

            # TODO: if i_st or i_sp = _np.nan

            idx_base = _np.arange(i_sp - i_st)

            coeff = (signal[i_sp] - signal[i_st]) / len(idx_base)

            driver_base = idx_base * coeff + signal[i_st]

            driver_no_peak[i_st:i_sp] = driver_base

        idx_grid = _np.arange(0, len(driver_no_peak) - 1, grid_size * fsamp)
        idx_grid = _np.r_[idx_grid, len(driver_no_peak) - 1]

        driver_grid = _UnevenlySignal(driver_no_peak, idx_grid, fsamp, "dEDA", signal.get_start_time(),
                                      signal.get_metadata())

        tonic = driver_grid.to_evenly(kind='cubic')

        phasic = signal - tonic

        return phasic, tonic, driver_no_peak

    _params_descriptors = {
        'delta': _Par(2, float, 'Minimum amplitude of the peaks in the driver', 0, lambda x: x > 0),
        'grid_size': _Par(0, int, 'Sampling size of the interpolation grid in seconds', 1, lambda x: x > 0),
        'pre_max':
            _Par(1, float,
                 'Duration (in seconds) of interval before the peak that is considered to find the start of the peak',
                 2, lambda x: x > 0),
        'post_max':
            _Par(1, float,
                 'Duration (in seconds) of interval after the peak that is considered to find the start of the peak',
                 2, lambda x: x > 0)
    }
