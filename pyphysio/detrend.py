#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 14:59:35 2020

@author: bizzego
"""
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import linregress
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.filters.filtertools import recursive_filter
from statsmodels.tsa.ar_model import AutoReg

LAG = 20

#%%
#x = arima.sim(list(order = c(1,1,0), ar = 0.7), n = 200) #Generates random data from ARIMA(1,1,0). This will generate a new data set for each call.
#z = ts.intersect(x, lag(x,-3), lag(x,-4)) #Creates a matrix z with columns, xt, xt-3, and xt-4
#y = 15+0.8*z[,2]+1.5*z[,3] #Creates y from lags 3 and 4 of randomly generated x
x = np.loadtxt('/home/bizzego/tmp/signal_x.txt', delimiter = ',')
y = np.loadtxt('/home/bizzego/tmp/signal_y.txt', delimiter = ',')

#%%

#ccf(z[,1],y,na.action = na.omit) #CCF between x and y
ccf = np.correlate(x, y, mode='full')

plt.stem(ccf[200-LAG:200+LAG])

#%%
#acf(x)
plot_acf(x, lags=22)

#%%
#ar1model = arima(x, order = c(1,1,0))
model = ARIMA(x, order=(1,1,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())

#%%
#pwx=ar1model$residuals
pwx=model_fit.resid

#%%
newpwy = np.convolve(y, [1, -1.7445, 0.7445], mode='full')

#%%
ccf = np.correlate(pwx, newpwy, mode='full')
plt.plot(ccf)

plot_acf(newpwy)

#%%
model = AutoReg(x, lags=5)
model_fit = model.fit()

coeffs = -model_fit.params[1:]

#%%
pwx = model_fit.resid

plot_acf(pwx)

#%%
newpwy = np.convolve(y, [1]+list(coeffs), mode='full')
plot_acf(newpwy)

#%%
ccf = np.correlate(pwx, newpwy, mode='full')

plt.stem(ccf[200-LAG:200+LAG])

