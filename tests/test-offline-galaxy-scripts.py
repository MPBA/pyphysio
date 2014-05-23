from galaxy.GalaxyHRVAnalysis import *
from galaxy.GalaxyLoadRR import *
from galaxy.GalaxyFilter import *
from galaxy.GalaxyNormalizeRR import *
from galaxy.GalaxyLinearTimeWindows import *
from pyHRV import get_available_indexes

hrv_list = get_available_indexes()
in_file = "../z_data/A05.txt"
rr_file = "rr.ibi"
out_file = "indexes.txt"
win_file = "wins.win"
GalaxyLoadRR(input=in_file, output=rr_file, data_type='rr', column=None).execute()
GalaxyFilter(input=rr_file, output=rr_file, column=None).execute()
GalaxyLinearTimeWindows(input=rr_file, output=win_file, step=20, width=40).execute()
GalaxyNormalizeRR(input=rr_file, output=rr_file, column=None, norm_mode="mean_sd").execute()
print GalaxyHRVAnalysis(input=rr_file, output=out_file, input_w=win_file, column=None, indexes=hrv_list).execute()
