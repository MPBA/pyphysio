import numpy as np
from pyphysio.pyPhysio.PhUI import PhUI

__author__ = 'aleb'


class PeakDetection(Estimator):
    @classmethod
    def algorithm(cls, data, params):
        PhUI.a('pkd_delta' in params, "The parameter 'pkd_delta is needed in order to perform a peak detection.")
        try:
            return PeakDetection._peak_detection(data, params['pkd_delta'], params['pkd_times'])
        except ValueError as e:
            PhUI.a(False, e.message)

    @staticmethod
    def _peak_detection(data, delta, times=None):
        """
        Detects peaks in the signal assuming the specified delta.
        @param data: Array of the values.
        @param delta: Differential threshold.
        @param times: Array of the times.
        @return: Tuple of lists: (max_t, min_t, max_v, min_v)
        @rtype: (list, list, list, list)
        @raise ValueError:
        """
        max_i = []
        min_i = []
        max_v = []
        min_v = []

        if times is None:
            times = np.arange(len(data))

        data = np.asarray(data)

        if len(data) != len(times):
            raise ValueError('Input vectors v and x must have same length')

        if not np.isscalar(delta):
            raise ValueError('Input argument delta must be a scalar')

        if delta <= 0:
            raise ValueError('Input argument delta must be positive')

        mn, mx = np.Inf, -np.Inf
        mn_pos, mx_pos = np.NaN, np.NaN

        look_for_max = True

        for i in np.arange(len(data)):
            this = data[i]
            if this > mx:
                mx = this
                mx_pos = times[i]
            if this < mn:
                mn = this
                mn_pos = times[i]

            if look_for_max:
                if this < mx - delta:
                    max_v.append(mx)
                    max_i.append(mx_pos)
                    mn = this
                    mn_pos = times[i]
                    look_for_max = False
            else:
                if this > mn + delta:
                    min_v.append(mn)
                    min_i.append(mn_pos)
                    mx = this
                    mx_pos = times[i]
                    look_for_max = True

        return max_i, min_i, max_v, min_v