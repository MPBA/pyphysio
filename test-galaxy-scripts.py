from Galaxy.calculate_HRV_indexes import *
from Galaxy.load_RR_data import *

hrv_list = np.ones(27)
GalaxyLoadRR(input="A05.txt", output="A05.ibi")
HRVAnalysis = GalaxyHRVAnalysis(input="A05.ibi", output="test2.txt", indexes=hrv_list)
hrv_data = HRVAnalysis.execute()

print hrv_data
