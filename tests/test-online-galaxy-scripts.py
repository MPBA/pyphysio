from pyPhysio.galaxy.GalaxyOnLineHRVAnalysis import GalaxyOnLineHRVAnalysis

__author__ = 'AleB'

from pyPhysio.Files import load_ds_from_csv_column
from pyPhysio.indexes import get_available_online_indexes
from pyPhysio.indexes.SupportValuesCollection import SupportValuesCollection

print(get_available_online_indexes())

a = load_ds_from_csv_column("../z_data/A05.txt")
s = SupportValuesCollection(indexes=get_available_online_indexes())
for i in a:
    g = GalaxyOnLineHRVAnalysis(indexes=get_available_online_indexes(), state=s, value=i)
    v, x = g.execute()
    print(v)
