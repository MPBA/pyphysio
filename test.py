import numpy as np

np.random.seed()

t=np.arange(0,10*np.pi,0.01*np.pi)
RRraw=np.random.uniform(500,1500,1000)+100*np.sin(t)

RRseries=DataSeries(RRraw)

TD_ind, TD_lab = RRAnalysis.TD_indexes(RRseries)
POIN_ind, POIN_lab = RRAnalysis.POIN_indexes(RRseries)

freq,spec=FFTCalc._calculate_data(RRseries,0.001)

print(TD_lab)
print(POIN_lab)