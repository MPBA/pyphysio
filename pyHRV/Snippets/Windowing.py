__author__ = 'AleB'
from pyHRV.windowing import *
from pyHRV import DataSeries

data = DataSeries()

## Linear Windows Generator
# Generates a linear-index set of windows (b+i*s, b+i*s+w).
wins1 = LinearWinGen(begin=0, step=20, width=40, data=data, end=100)
wins2 = LinearWinGen(0, 20, 40)

## Linear Time Windows Generator
# Generates a linearly-timed set of Time windows (b+i*s, b+i*s+w).
# Here the begin time is
wins3 = LinearTimeWinGen(begin=0, step=20, width=40, data=data, end=100)
wins4 = LinearTimeWinGen(0, 20, 40, data)

wm = WindowsMapper(rr, wg, [Td.Mean])
ws = WindowsMapper(rr, wg, [Td.SD])
wx = WindowsMapper(rr, wg, [Td.PNN25])
wm.compute_all()
ws.compute_all()
wx.compute_all()
print wm.results
print ws.results
print wx.results
