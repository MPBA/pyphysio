from Galaxy.calculate_HRV_indexes import *
from Galaxy.load_RR_data import *
import numpy as np

hrv_list = np.ones(27)
GalaxyLoadRR(input="A05.txt", output="A05.ibi", column="IBI").execute()
print GalaxyHRVAnalysis(input="A05.ibi", output="test2.txt", indexes=hrv_list).execute()

