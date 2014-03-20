from Galaxy.calculate_HRV_indexes import *

hrv_list = np.ones(27)
HRVAnalysis = GalaxyHRVAnalysis()
hrv_data = HRVAnalysis.execute('A05.txt', hrv_list)
