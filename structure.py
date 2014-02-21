##############
# Data Classes
##############
class DataSeries(object): #extend Series of pandas library???
    def __init__():

    # iterator
    # http://stackoverflow.com/questions/19151/build-a-basic-python-iterator
    def __iter__(self):

    def next(self):

    def load(self): #load from file

    def save(self): #save to file

    ### functions to manage cache

class PhysioData(DataSeries):

    data # values of series

    fsampling = # Hz
    resolution = #
    device = ''
    acquisition_date = time()
    subject = ''

    def load(self): #load from file

    def save(self): #save to file

class RRData(PhysioData):

class HeartData(PhysioData):
    def returnRR(self): # peak detection and RRfiltering
        return(RRData)

#class SKData(Physiodata):

#class ACCData(PhysioData):

class TimeStampData(DataSeries):
    def load(self): #load from file

    def save(self): #save to file

class WindowData(TimeStampData):

##############
# Cacheable
##############

class CacheableDataCalc(object):

class FFTCalc(CacheableDataCalc):

class RRDiff(CacheableDataCalc):

# class Interpolate(CacheableDataCalc): #forse no

##############
# Analysis Classes - STATIC
##############

class DataAnalysis(object):

class RRAnalysis(DataAnalysis):

class RRFiltering(RRAnalysis):

### more utility classes

##############
# Indexes Classes
##############

class Index():
    value=
    support_value= (Cacheable) #for online processing
    nsamples=

    def calculate(self, DataSeries, ind_start=0, ind_end=-1):

class TDIndex(): # compute TD index

class RRmean, HRmean, RRSTD, HRSTD (TDIndex):

class pNNx, NNx, RMSSD, SDSD(TDIndex):
    # uses RRDiff(CacheableDataCalc)

class ...

class FDIndex(): #calculate FD index
    def _interpolate(self, fsamp):

    def _estimatePSD(self, fsamp, method):

    def calculate(self):
    # uses FFTCalc(CacheableDataCalc)

class VLF, LF, HF(): #calculate absolute, peak, %1, %2 (4 indexes)
    # uses FFTCalc
    # should be Cacheable? need its value to calculate other indexes

class Total():
    # uses FFTCalc
    # should be Cacheable? need its value to calculate other indexes

class LFHF()
    # uses FFTCalc
    # should be Cacheable? need its value to calculate other indexes

class POIndex(): # calculate poincare' index
class NLIndex(): ...