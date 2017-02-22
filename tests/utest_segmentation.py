from __future__ import division # divisione sensata
import numpy as np # strumenti matematici
import pandas as pd # importare dati
import os # navigare nel computer
import matplotlib.pyplot as plt # strumenti per fare i grafici
import pyphysio as ph


FS_BVP = 64

os.chdir('/home/andrea/Trento/DATI/yagmur_2016/parentsTD/SUB246/SUB246/Empatica E4/')

#EVENLY
bvp_data=np.array(pd.read_csv('BVP.csv', sep=';', names=['timestamp', 'signal'], usecols=[0,1], skiprows=1))
bvp = ph.EvenlySignal(bvp_data[:,1], FS_BVP, 'BVP')

bvp_segment = bvp.segment_time(300,400)


#UNEVENLY
ibi = ph.BeatFromBP(bpm_max=120)(bvp.resample(1024))

ibi_segment = ibi.segment_time(300,400)


ax1 = plt.subplot(211)
bvp.plot()
bvp_segment.plot()
plt.vlines(ibi.get_times(), np.min(bvp), np.max(bvp))

plt.subplot(212, sharex = ax1)
ibi.plot('o-')
ibi_segment.plot('o-')

#EVENTSSIGNAL
sounds_data = pd.read_csv('/home/andrea/Trento/DATI/yagmur_2016/parentsTD/SUB246/20150802_182543_246_SUB/snd.csv', sep = '\t')
tags = ph.EventsSignal(sounds_data.name, sounds_data.timestamp, 1000, sounds_data.timestamp/1000, 'snd', 0)

out_signal = bvp.segment_time(300,400)