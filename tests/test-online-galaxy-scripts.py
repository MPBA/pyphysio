from pyHRV.galaxy.GalaxyOnLineHRVAnalysis import GalaxyOnLineHRVAnalysis

__author__ = 'AleB'

from pyHRV.Files import load_rr
from pyHRV.indexes import get_available_online_indexes
from pyHRV.indexes.SupportValuesCollection import SupportValuesCollection

print(get_available_online_indexes())

a = load_rr("../z_data/A05.txt")
s = SupportValuesCollection(indexes=get_available_online_indexes())
for i in a:
    g = GalaxyOnLineHRVAnalysis(indexes=get_available_online_indexes(), state=s, value=i)
    v, s = g.execute()
    print(v)
