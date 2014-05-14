import numpy as np

import matplotlib.pyplot as pl

from pyHRV.Files import *
from pyHRV.Filters import *


if __name__ == '__main__':
    a = load_rr_from_bvp("../z_data/BVP.txt", ';')
    pl.show(pl.plot(np.cumsum(a), a))
    b = RRFilters.normalize_mean_sd(a)
    pl.show(pl.plot(np.cumsum(a), b))
    print b


### TODO: WIP

hist, bins = np.histogram(RR, 100)
MAX = np.max(hist)
#---
hist_left = np.array(hist[0:np.argmax(hist)])
L = len(hist_left)
hist_right = np.array(hist[np.argmax(hist):-1])
R = len(hist_right)

Y_left = np.linspace(0, MAX, len(hist_left))

min = np.Inf
pos = 0
for i in range(len(hist_left) - 1):
    currmin = np.sum((hist_left - Y_left) ** 2)
    if currmin < min:
        min = currmin
        pos = i
    Y_left[i] = 0
    Y_left[i + 1:] = np.linspace(0, MAX, L - i - 1)

N = bins[pos - 1]

Y_right = np.linspace(MAX, 0, len(hist_right))
min = np.Inf
pos = 0
for i in range(len(hist_right), 1, -1):
    currmin = np.sum((hist_right - Y_right) ** 2)
    if currmin < min:
        min = currmin
        pos = i
    Y_right[i - 1] = 0
    Y_right[0:i - 2] = np.linspace(MAX, 0, i - 2)

M = bins[np.argmax(hist) + pos + 1]

TINN = M - N
