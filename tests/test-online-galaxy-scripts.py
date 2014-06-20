from galaxy.GalaxyOnLineHRVAnalysis import GalaxyOnLineHRVAnalysis

__author__ = 'AleB'

from pyHRV.Files import load_rr_from_bvp
from pyHRV import get_available_online_indexes
from pyHRV.indexes.SupportValuesCollection import SupportValuesCollection

a = load_rr_from_bvp("../z_data/BVP.txt", sep=';')
s = SupportValuesCollection(indexes=get_available_online_indexes())
for i in a:
    g = GalaxyOnLineHRVAnalysis(indexes=get_available_online_indexes(), stste=s, value=i)
    print(g.execute())
