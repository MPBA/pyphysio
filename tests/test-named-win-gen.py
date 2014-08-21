__author__ = 'AleB'

import os

import pandas as pd

from pyHRV.Files import *
from pyHRV.windowing import NamedWinGen, WindowsMapper
from pyHRV import DataSeries
import pyHRV.indexes.TDIndexes
import pyHRV.indexes.FDIndexes
from pyHRV.PyHRVSettings import PyHRVDefaultSettings as Ps


def test(f):
    print f

    lab, ibi = load_pd_from_excel_column(f, 2, 3)

    ds = DataSeries(ibi, labels=lab)
    ws = NamedWinGen(ds, include_baseline_name="baseline")

    mm = WindowsMapper(ds, ws, pyHRV.indexes.TDIndexes.__all__ + pyHRV.indexes.FDIndexes.__all__)
    mm.compute_all()
    df = pd.DataFrame(mm.results)
    df.columns = mm.labels
    df.to_csv(os.path.dirname(f) + "/results/" + os.path.basename(f) + ".results.csv", Ps.load_csv_separator,
              index=False)


def main():
    test_dir('../z_data/ECG_nonfathers/')
    test_dir('../z_data/ECG_fathers/')


def test_dir(d):
    if not os.path.exists(d + "/results/"):
        os.makedirs(d + "/results/")
    for a in os.listdir(d):
        if os.path.isfile(d + a):
            test(d + a)


if __name__ == "__main__":
    main()
