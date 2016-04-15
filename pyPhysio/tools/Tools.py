import numpy as _np

from ..filters.Filters import ConvolutionalFilter as _ConvFlt
from ..PhUI import PhUI as _ph
from ..BaseTool import Tool as _Tool
from ..Signal import UnevenlySignal as _Unev

"""
Tools are generic operations that take as input a SIGNAL (or numpy array?) and gives as output one or more np.array.
"""


# TODO: len(signal) : can I use signal.duration? Signal.get_duration() returns the time length in [s]

class PeakDetection(_Tool):
    """
    Estimate the maxima and the minima in the signal (in particular for periodic signals).

    Parameters
    ----------
    deltas= : nparray or float
        The threshold for the detection of the peaks. If array it should have the same length of the signal.
    refractory= : int
        Number of samples to skip after detection of a peak
    start_max= : boolean
        Whether to start looking for a max.

    Returns
    -------
    maxs : nparray
        Array containing indexes (first column) and values (second column) of the maxima
    mins : nparray
        Array containing indexes (first column) and values (second column) of the minima
    """

    @classmethod
    def algorithm(cls, signal, params):
        delta = params['delta']
        refractory = params['refractory']
        start_max = params['start_max']
        if _np.shape(delta) == ():
            deltas = _np.repeat(delta, len(signal))

        # TODO: check added code
        # ---
        else:
            from ..estimators import Diff
            deltas = Diff()(signal)
        mins = []
        maxs = []
        # ---

        # initialization
        mn_candidate, mx_candidate = _np.Inf, -_np.Inf
        mn_pos_candidate, mx_pos_candidate = _np.nan, _np.nan

        look_for_max = start_max

        i_activation_max = 0
        i_activation_min = 0

        i = 0
        while i <= len(signal) - 1:
            this = signal[i]
            delta = deltas[i]

            if this > mx_candidate:
                mx_candidate = this
                mx_pos_candidate = i
            if this < mn_candidate:
                mn_candidate = this
                mn_pos_candidate = i

            if look_for_max:
                if i >= i_activation_max and this < mx_candidate - delta:  # new max
                    maxs.append((mx_pos_candidate, mx_candidate))
                    i_activation_max = i + refractory

                    mn_candidate = this
                    mn_pos_candidate = i

                    look_for_max = False
            else:
                if i >= i_activation_min and this > mn_candidate + delta:  # new min
                    mins.append((mn_pos_candidate, mn_candidate))
                    i_activation_min = i + refractory

                    mx_candidate = this
                    mx_pos_candidate = i

                    look_for_max = True
            i += 1
        return _np.array(maxs), _np.array(mins)

    @classmethod
    def check_params(cls, params):
        if 'delta' not in params:
            # default = 0 # GRAVE
            pass
        if 'refractory' not in params or params['refractory'] < 0:
            # default = 1 # OK
            pass
        if 'startMax' not in params:
            # default = True # OK
            pass
        return params


class Range(_Tool):
    """
    Estimate the local range of the signal

    Parameters
    ----------
    winstep : int
        The increment indexes to start the next window
    winlen : int
        The dimension of the window
    smooth : boolean
        Whether to convolve the result with a gaussian window

    Returns
    -------
    deltas : nparray
        Result of estimation of local range
    """

    @classmethod
    def algorithm(cls, signal, params):
        win_len = params['win_len']
        win_step = params['win_step']
        smooth = params['smooth']

        windows = _np.arange(0, len(signal) - win_len, win_step)
        deltas = _np.zeros(len(signal))
        # TODO: check added code
        # ---
        curr_delta = None
        # ---
        for start in windows:
            portion_curr = signal[start: start + win_len]
            curr_delta = (_np.max(portion_curr) - _np.min(portion_curr))
            deltas[start:start + win_len] = curr_delta
        # ---
        start = windows[-1]
        # ---
        deltas[start + win_len:] = curr_delta

        if smooth:
            deltas = _ConvFlt(irftype='gauss', N=win_len * 2, normalize=True)(deltas)  # TODO: check sintax

        return deltas

    @classmethod
    def check_params(cls, params):
        if not 'win_len' in params:
            # default = None # GRAVE
            pass
        if not 'win_step' in params:
            # default = None # GRAVE
            pass
        if not 'smooth' in params:
            # default = True # OK
            pass
        return params


class PSD(_Tool):
    """
    """

    @classmethod
    def algorithm(cls, signal, params):
        pass

    @classmethod
    def check_params(cls, params):
        return params


class Energy(_Tool):
    """
    Estimate the local energy of the signal

    Parameters
    ----------
    winstep : int
        The increment indexes to start the next window
    winlen : int
        The dimension of the window
    smooth : boolean
        Whether to convolve the result with a gaussian window

    Returns
    -------
    energy : nparray
        Result of estimation of local energy
    """

    @classmethod
    def algorithm(cls, signal, params):
        winlen = params['win_len']
        winstep = params['win_step']
        smooth = params['smooth']

        windows = _np.arange(0, len(signal) - winlen, winstep)

        energy = []
        curr_energy = None
        for start in windows:
            portion_curr = signal[start: start + winlen]
            curr_energy = _np.sum(_np.power(portion_curr, 2)) / len(portion_curr)
            energy.append(curr_energy)
        energy.append(curr_energy)
        energy.insert(0, energy[0])

        idx_interp = _np.r_[0, windows + round(winlen / 2), len(signal)]
        energy = _np.array(energy)
        # TODO: assumed ", 1," was the wanted fsmp
        # WAS: energy_out = flt.interpolate_unevenly(energy, idx_interp, 1, kind='linear')
        energy_out = _Unev(energy, idx_interp, len(idx_interp) + 1, 1).to_evenly(kind='linear').get_y_values()

        if smooth:
            # TODO: check sintax
            energy_out = _ConvFlt(irftype='gauss', N=winlen * 2, normalize=True)(energy_out)

        return energy_out

    @classmethod
    def check_params(cls, params):
        if 'win_len' not in params:
            # default = None # GRAVE
            pass
        if 'win_step' not in params:
            # default = None # GRAVE
            pass
        if 'smooth' not in params:
            # default = True # OK
            pass
        return params


class Maxima(_Tool):
    """
    """

    @classmethod
    def algorithm(cls, signal, params):
        pass

    @classmethod
    def check_params(cls, params):
        return params


class Minima(_Tool):
    """
    """

    @classmethod
    def algorithm(cls, signal, params):
        pass

    @classmethod
    def check_params(cls, params):
        return params


class CreateTemplate(_Tool):
    """
    """

    @classmethod
    def algorithm(cls, signal, params):
        pass

    @classmethod
    def check_params(cls, params):
        return params


class OptimizeBateman(_Tool):
    """
    """

    @classmethod
    def algorithm(cls, signal, params):
        pass

    @classmethod
    def check_params(cls, params):
        return params

    @staticmethod
    def _response_energy(driver, fsamp, delta):
        """
        Estimates the phasic and tonic components of a driver.

        It uses a detection algorithm based on the derivative of the driver.

        Parameters
        ----------
        driver : nparray
            The driver signal
        fsamp : float
            The sampling frequency
        delta : float
            Minimum amplitude of the peaks in the driver

        Returns
        -------
        response : float
            Response energy computed on the driving signal

        Raises
        ------
        RuntimeWarning: if no eligible peaks are found to compute the response energy

        Notes
        -----
        See paper ...
        """
        maxs, mins = tll.peakdet(driver, delta)

        if len(maxs) != 0:
            idx_maxs = maxs[:, 0]
        else:
            _ph.w('Unable to find peaks in driver signal for computation of Energy. Returning Inf')
            return 10000

        # STAGE 1: select maxs distant from the others
        diff_maxs = _np.diff(_np.r_[idx_maxs, len(driver) - 1])
        th_diff = 15 * fsamp

        # TODO: select th such as to have enough maxs, e.g. diff_maxs_tentative = np.median(diff_maxs)
        idx_selected_maxs = _np.where(diff_maxs > th_diff)[0]
        selected_maxs = idx_maxs[idx_selected_maxs]

        if len(selected_maxs) != 0:
            energy = 0
            for idx_max in selected_maxs:
                driver_portion = driver[idx_max:idx_max + 15 * fsamp]

                half = len(driver_portion) - 5 * fsamp

                y = driver_portion[half:]
                diff_y = _np.diff(y)
                th_75 = _np.percentile(diff_y, 75)
                th_25 = _np.percentile(diff_y, 25)

                idx_sel_diff_y = _np.where((diff_y > th_25) & (diff_y < th_75))[0]
                diff_y_sel = diff_y[idx_sel_diff_y]

                #            A2, B = _find_slope(y, half)
                mean_s = tll.bootstrap_estimation(diff_y_sel, _np.mean, N=100, k=0.5)

                mean_y = tll.bootstrap_estimation(y, _np.median)
                b_mean_s = mean_y - mean_s * (half + (len(driver_portion) - half) / 2)

                #            line_A2 = A2 * np.arange(len(driver_portion)) + B
                line_mean_s = mean_s * _np.arange(len(driver_portion)) + b_mean_s

                #            plt.figure()
                #            plt.plot(diff_y)
                #            plt.plot(idx_sel_diff_y, diff_y_sel, 'o')
                #            plt.hlines(mean_s, 0, len(diff_y))
                #            plt.plot(driver_portion)
                #            plt.plot(np.arange(half, half+len(y)), y, 'r', linewidth=1.2)
                #            plt.plot(line_mean_s, 'k')
                #            plt.show()

                driver_detrended = driver_portion - line_mean_s
                #            if driver_detrended[0] < 0:
                #                print('driver0')
                driver_detrended /= _np.max(driver_detrended)
                energy_curr = (1 / fsamp) * _np.sum(driver_detrended[fsamp:] ** 2) / (len(driver_detrended) - fsamp)

                #                driver_portion = driver_detrended / driver_portion[0]
                #                idx_negative = np.where(driver_portion<0)[0]
                #
                #                if len(idx_negative)!=0:
                #                    idx_t1 = idx_negative[0]
                #                    driver_t1_on = driver_portion[idx_t1:]
                #                    energy_curr = (1/fsamp) * np.sum(driver_t1_on**2) / len(driver_t1_on)
                #                else:
                #                    energy_curr = (1/fsamp) * np.sum(driver_portion**2) / len(driver_portion)
                #            else:
                energy += energy_curr
            return energy

        else:
            _ph.w('Peaks found but too near. Returning Inf')
            return 10000
            #            else:
            #                driver_portion = driver_detrended / abs(driver_detrended[0])
            #                energy_curr = (1/fsamp)*np.sum(driver_detrended**2)/len(driver_detrended)

    @staticmethod
    def _loss_function(par_bat, signal, fsamp, delta=20, verbose=False):
        """
        Computes the loss for optimization of Bateman parameters.

        Parameters
        ----------
        par_bat : list
            Bateman parameters to be optimized
        signal : nparray
            The EDA signal
        fsamp : float
            The sampling frequency
        delta : float
            Minimum amplitude of the peaks in the driver
        verbose : boolean
            Print Bateman parameters and loss at each evaluation

        Returns
        -------
        loss : float
            The computed loss
        """
        t1_min = 0.04
        t2_max = 15
        if par_bat[0] < t1_min or par_bat[1] > t2_max or par_bat[0] >= par_bat[1]:
            return 10000

        driver = estimate_driver(signal, fsamp, par_bat)
        loss = OptimizeBateman._response_energy(driver, fsamp, delta)
        if verbose:
            print(par_bat, loss)
        return loss


class BootstrapEstimation(_Tool):
    """
    Perform a bootstrapped estimation of given statistical indicator
    
    Parameters
    ----------
    func : numpy function
        The estimator. Must accept data as input
    N : int
        Number of iterations
    k : float (0-1)
        Portion of data to be used at each iteration
    
    Returns
    -------
    estim : float
        The bootstrapped estimate
    """

    @classmethod
    def algorithm(cls, signal, params):
        l = len(signal)  # TODO: can I use signal.duration? Signal.get_duration() gives the time length in [s]
        niter = params['niter']
        func = params['func']
        k = params['k']

        estim = []
        for i in range(niter):
            ixs = _np.arange(l)
            ixs_p = _np.random.permutation(ixs)
            sampled_data = signal[ixs_p[:round(k * l)]]
            curr_est = func(sampled_data)
            estim.append(curr_est)
        return _np.mean(estim)

    @classmethod
    def check_params(cls, params):
        if 'func' not in params:
            # default = _np.mean # GRAVE
            pass
        if 'niter' not in params:
            # default = 100 # OK
            pass
        if 'k' not in params:
            # default = 0.5 # OK
            pass
        return params
