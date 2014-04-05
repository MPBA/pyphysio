__author__ = 'AleB'

from pyHRV.windowing import *
from pyHRV.Files import *
from pyHRV.indexes import TDIndexes as Td


def main():
    rr = load_rr_data_series("../z_data/A05.txt")
    wg = LinearWinGen(rr, 0, 20, 40)
    wm = WindowsMapper(rr, wg, Td.Mean)
    wm.compute()
    print wm.results


if __name__ == "__main__":
    main()
