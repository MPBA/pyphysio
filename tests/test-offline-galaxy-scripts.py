from galaxy.GalaxyHRVAnalysis import *
from galaxy.GalaxyLoadRR import *
from galaxy.GalaxyFilter import *
from galaxy.GalaxyNormalizeRR import *
from pyHRV import get_available_indexes

hrv_list = get_available_indexes()
in_file = "../z_data/A05.txt"
rr_file = "rr.ibi"
out_file = "indexes.txt"
GalaxyLoadRR(input=in_file, output=rr_file, data_type='rr', column=None).execute()
GalaxyFilter(input=rr_file, output=rr_file, column=None).execute()
GalaxyNormalizeRR(input=rr_file, output=rr_file, column=None, norm_mode="mean").execute()
print GalaxyHRVAnalysis(input=rr_file, output=out_file, column=None, indexes=hrv_list).execute()
