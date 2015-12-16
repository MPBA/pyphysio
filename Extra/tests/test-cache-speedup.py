__author__ = 'AleB'
from time import time

from pyPhysio.galaxy.GalaxyHRVAnalysis import *
from pyPhysio.Files import load_ds_from_csv_column
from pyPhysio.features import get_available_indexes


inp = "../z_data/A05.txt"
ds = load_ds_from_csv_column(inp)
i = get_available_indexes()
t = time()
for x in i:
    print x, "\t", getattr(pyHRV, x)(ds).value
c = time() - t
print "-----------Cached:", c
ds = pd.Series(ds)
t = time()
for x in i:
    print x, "\t", getattr(pyHRV, x)(ds).value
u = time() - t
print "-----------Cached:", c
print "-----------Non cached:", u
print "-----------Ratio:", u / c * 100 - 100, "%"
