__author__ = 'Andrea'
import numpy as np
from scipy import interpolate


def power(spec, freq, min_freq, max_freq):
    """Returns the power calculated in the specified band of the spec-freq spectrum
    @param spec: power values
    @param freq: frequency values
    @param min_freq: lower band bound
    @param max_freq: higher band bound
    @return: power in band
    """
    band = np.array([spec[i] for i in range(len(spec)) if min_freq <= freq[i] < max_freq])
    return np.sum(band) / len(spec)


def interpolate_rr(rr, interp_freq):
    """Returns as a tuple the interpolated RR and BT arrays
    @param rr:
    @param interp_freq:
    @return: (RR interpolated, BT interpolated)
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


def template_interpolation(x, t, freq, template=None):
    if template is None:
        template = 0.5 * np.cos(np.arange(0, 1.01, 0.01) * np.pi) + 0.5

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

    t_output = np.arange(t[0], t[-1], 1 / freq)

    f = interpolate.interp1d(t_out, x_out, 'linear')
    x_output = f(t_output)
    return x_output, t_output


def build_takens_vector(x, m):
    #creo embedded matrix
    #righe = Uj
    n = len(x)
    num = n - m + 1
    emb = np.zeros([num, m])
    for i in xrange(num):
        emb[i, :] = x[i:i + m]
    return emb


def peak_detection(v, delta, x=None):
    max_tab = []
    min_tab = []

    if x is None:
        x = np.arange(len(v))

    v = np.asarray(v)

    if len(v) != len(x):
        raise ValueError('Input vectors v and x must have same length')

    if not np.isscalar(delta):
        raise ValueError('Input argument delta must be a scalar')

    if delta <= 0:
        raise ValueError('Input argument delta must be positive')

    mn, mx = np.Inf, -np.Inf
    mn_pos, mx_pos = np.NaN, np.NaN

    look_for_max = True

    for i in np.arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mx_pos = x[i]
        if this < mn:
            mn = this
            mn_pos = x[i]

        if look_for_max:
            if this < mx - delta:
                max_tab.append((mx_pos, mx))
                mn = this
                mn_pos = x[i]
                look_for_max = False
        else:
            if this > mn + delta:
                min_tab.append((mn_pos, mn))
                mx = this
                mx_pos = x[i]
                look_for_max = True

    return np.array(max_tab), np.array(min_tab)
