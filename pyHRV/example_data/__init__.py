__author__ = 'AleB'

from pandas import read_csv
from os.path import join, dirname, abspath

from pyHRV import DataSeries


name_d = DataSeries.__name__.lower()
name_r = "results"


def load_example(f):
    n = dirname(abspath(f))
    uu = read_csv(join(n, "RD.txt"), sep="\t")
    d, r = DataSeries(data=uu["IBI"], labels=uu["label"]), read_csv(join(n, "results.csv"), sep="\t")
    return d, r

import Test1
import Test2
import Test3
