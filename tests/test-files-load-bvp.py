from pyHRV.indexes.TDIndexes import TINN, Triang
from pyHRV.Files import *
from pyHRV.Filters import *


if __name__ == '__main__':
    a = load_rr_from_bvp("../z_data/BVP.txt", ';')
    b = RRFilters.filter_out_layers(a)
    print "Triang: ", Triang(b).value
    print "TINN:   ", TINN(b).value

