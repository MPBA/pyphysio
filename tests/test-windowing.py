__author__ = 'AleB'

from pyHRV.windowing import *
from pyHRV.Files import *
from pyHRV.indexes import TDFeatures as Td


def main():
    rr = load_ds_from_csv_column("../z_data/D01.txt")
    wg = LinearWinGen(0, 20, 40, rr)
    wm = WindowsIterator(rr, wg, [Td.Mean])
    ws = WindowsIterator(rr, wg, [Td.SD])
    wx = WindowsIterator(rr, wg, [Td.PNN25])
    wm.compute_all()
    ws.compute_all()
    wx.compute_all()
    print wm.results
    print ws.results
    print wx.results


if __name__ == "__main__":
    main()
