from galaxy.HRV_analysis_Galaxy import *

hrv_list = np.ones(27)
HRVAnalysis = GalaxyHRVAnalysis()
hrv_data = HRVAnalysis.execute(input='A05', indexes=hrv_list)
