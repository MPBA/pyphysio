import numpy as np
from scipy import interpolate


def power(spec, freq, min_freq, max_freq):
    #returns power in band
    band = np.array([spec[i] for i in range(len(spec)) if min_freq <= freq[i] < max_freq])
    return np.sum(band) / len(spec)


def interpolate_rr(rr, interp_freq):
    # returns cubic spline interpolated array with sample rate = interp_freq
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


def smooth_triangle(data, degree):
    triangle = np.array(range(degree) + [degree] + range(degree)[::-1]) + 1
    smoothed = data[0:degree]
    for i in range(degree, np.size(data) - degree * 2):
        point = data[i:i + len(triangle)] * triangle
        smoothed = np.append(smoothed, sum(point) / sum(triangle))
    smoothed = np.insert(smoothed, -1, data[-(degree * 2):])
    return smoothed


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


def build_takens_vector(x, m):
    #creo embedded matrix
    #righe = Uj
    n = len(x)
    num = n - m + 1
    emb = np.zeros([num, m])
    for i in xrange(num):
        emb[i, :] = x[i:i + m]
    return emb
