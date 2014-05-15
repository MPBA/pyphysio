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
