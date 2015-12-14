# coding=utf-8
__author__ = 'AleB'
import numpy as np
from scipy import interpolate
from pandas import TimeSeries
from pyPhysio.PyHRVSettings import MainSettings as Sett


def data_series_from_bvp(bvp, bvp_time, delta_ratio=Sett.import_bvp_delta_max_min_numerator,
                         filters=Sett.import_bvp_filters):
    """
    Loads an IBI (RR) data series from a BVP data set and filters it with the specified filters list.
    @param delta_ratio: delta parameter for the peak detection
    @type delta_ratio: float
    @param bvp: ecg values column
    @type bvp: Iterable
    @param bvp_time: ecg timestamps column
    @type bvp_time: Iterable
    @param filters: sequence of filters to be applied to the data (e.g. from IBIFilters)
    @return: Filtered signal DataSeries
    @rtype: DataSeries
    """
    delta = (max(bvp) - min(bvp)) / delta_ratio
    max_i, ii, iii, iv = peak_detection(bvp, delta, bvp_time)
    s = TimeSeries(np.diff(max_i) * 1000)
    for f in filters:
        s = f(s)
    s.meta_tag['from_type'] = "data_time-bvp"
    s.meta_tag['from_peak_delta'] = delta
    s.meta_tag['from_freq'] = np.mean(np.diff(bvp_time))
    s.meta_tag['from_filters'] = list(Sett.import_bvp_filters)
    return s


def data_series_from_ecg(ecg, ecg_time, delta=Sett.import_ecg_delta, filters=Sett.import_bvp_filters):
    """
    Loads an IBI (RR) data series from an ECG data set and filters it with the specified filters list.
    @param delta: delta parameter for the peak detection
    @type delta: float
    @param ecg: ecg values column
    @type ecg: Iterable
    @param ecg_time: ecg timestamps column
    @type ecg_time: Iterable
    @return: Filtered signal DataSeries
    @rtype: DataSeries
    """
    # TODO: explain delta
    max_tab, min_tab, ii, iii = peak_detection(ecg, delta, ecg_time)
    s = TimeSeries(np.diff(max_tab))
    for f in filters:
        s = f(s)
    s.meta_tag['from_type'] = "data_time-ecg"
    s.meta_tag['from_peak_delta'] = delta
    s.meta_tag['from_freq'] = np.mean(np.diff(ecg_time))
    s.meta_tag['from_filters'] = list(Sett.import_ecg_filters)
    return s


def derive_holdings(data, labels):
    ll = []
    tt = []
    ii = []
    ts = 0
    pre = None
    for i in xrange(len(labels)):
        if pre != labels[i]:
            ll.append(labels[i])
            tt.append(ts)
            ii.append(i)
            ts += data[i]
            pre = labels[i]
    return ll, tt, ii


def power(spec, freq, min_freq, max_freq):
    """
    Returns the power calculated in the specified band of the spec-freq spectrum
    """
    band = np.array([spec[i] for i in range(len(spec)) if min_freq <= freq[i] < max_freq])
    return np.sum(band) / len(spec)


def interpolate_ibi(rr, interp_freq):
    """
    Returns as a tuple the interpolated RR and BT arrays
    """
    step = 1.0 / interp_freq
    rr /= 1000
    rr = np.array(rr)
    bt = np.cumsum(rr)
    x_min = bt[0]
    x_max = bt[-1]
    bt = np.insert(bt, 0, 0)
    bt = np.append(bt, bt[-1] + 1)
    rr = np.insert(rr, 0, 0)
    rr = np.append(rr, rr[-1])
    tck = interpolate.splrep(bt, rr)
    bt_interp = np.arange(x_min, x_max, step)
    rr_interp = interpolate.splev(bt_interp, tck)
    return rr_interp, bt_interp


def template_interpolation(x, t, step, template=None):
    if template is None:
        template = np.square(np.cos(np.arange(0, 0.505, 0.005) * np.pi))

    x_old = x[0]
    t_old = t[0]

    x_out = np.array([])
    t_out = np.array([])

    for i in xrange(1, len(x)):
        x_curr = x[i]
        t_curr = t[i]

        x_template = template * (x_old - x_curr) + x_curr
        t_template = np.linspace(t_old, t_curr, 101)

        x_out = np.hstack((x_out, x_template, x_curr))
        t_out = np.hstack((t_out, t_template, t_curr))

        t_old = t_curr
        x_old = x_curr

    t_output = np.arange(t[0], t[-1], step)

    f = interpolate.interp1d(t_out, x_out, 'linear')
    x_output = f(t_output)
    return x_output, t_output


def ordered_subsets(x, n):
    num = len(x) - n + 1
    if num > 0:
        emb = np.zeros([num, n])
        for i in xrange(num):
            emb[i, :] = x[i:i + n]
        return emb
    else:
        return []


def peak_detection(data, delta, times=None):
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
