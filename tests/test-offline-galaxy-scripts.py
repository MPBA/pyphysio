from galaxy.GalaxyHRVAnalysis import *
from galaxy.GalaxyLoadRR import *
from pyHRV import get_available_indexes

hrv_list = get_available_indexes()
in_file = "z_data/A05.txt"
rr_file = "rr.ibi"
out_file = "indexes.txt"
GalaxyLoadRR(input=in_file, output=rr_file, column="IBI").execute()
print GalaxyHRVAnalysis(input=rr_file, output=out_file, indexes=hrv_list).execute()
