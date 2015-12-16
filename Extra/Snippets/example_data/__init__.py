__author__ = 'AleB'

from pandas import read_csv
from os.path import join, dirname, abspath

name_r = "results"


def load_example(f):
    n = dirname(abspath(f))
    uu = read_csv(join(n, "RD.txt"), sep="\t")
    d, r = Cache(data=uu["IBI"], labels=uu["label"]), read_csv(join(n, "results.csv"), sep="\t")
    return d, r


from Extra.Snippets.example_data import Test3
