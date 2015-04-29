from pyHRV.indexes.TDFeatures import TINN, Triang
from pyHRV.Files import load_ibi_from_bvp
from pyHRV.Filters import Filters


if __name__ == '__main__':
    a = load_ibi_from_bvp("../z_data/BVP.txt", sep=';')
    b = Filters.filter_outliers(a)
    print "Triang: ", Triang(b).value
    print "TINN:   ", TINN(b).value

