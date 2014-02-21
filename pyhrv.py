from __future__ import division
__author__ = 'ale'

from __builtin__ import staticmethod

from utility import *


from numpy import *

import spectrum as spct

from scipy import interpolate


# La struttura dati interna puo' essere di tipo arbitrario, lascio stare gli RR intanto
class DataSeries(object):
    """Data series class. Wraps a data structure and gives a cache support through CacheableDataCalc subclasses."""

    def __init__(self, data=None):
        """ Default constructor.
        :param data: Data to carry or None
        :return:
        """
        # lista dei dati originali
        if data is None:
            self._data = {}
        else:
            self._data = data
        # dizionario per i dati in cache
        self._cache = {}

    # proprieta' per i dati orig.
    @property
    def series(self):
        """ Internal data object
        """
        return self._data

    @series.setter
    def series(self, data):
        """ Flushes the cache and overwrites the internal list
        :param data: Data to set
        """
        self._data = data
        self.cache_clear()

    # libera la cache
    def cache_clear(self):
        """ Clears the cache and frees memory (GC?)
        """
        self._cache = {}

    # see class CacheableDataCalc
    def cache_check(self, calculator):
        """ Check if the cache contains valid calculator's data
        :type calculator: CacheableDataCalc
        :param calculator: CacheableDataCalc
        :return: If the cache is valid
        """
        assert calculator is CacheableDataCalc
        return calculator.cid() in self._cache

    def cache_invalidate(self, calculator):
        """
        :type calculator: CacheableDataCalc
        :param calculator: CacheableDataCalc
        """
        assert calculator is CacheableDataCalc
        if self.cache_check(calculator):
            del self._cache[calculator.cid()]

    def cache_pre_calc_data(self, calculator):
        """ Precalculates data and caches it
        :type calculator: CacheableDataCalc
        :param calculator: CacheableDataCalc
        """
        assert calculator is CacheableDataCalc
        # aggiungo alla cache
        self._cache[calculator.cid()] = calculator.get(self, False)
        return self._cache[calculator.cid()]

    def cache_get_data(self, calculator):
        """ Gets data from the cache if valid
        :type calculator: CacheableDataCalc
        :param calculator: CacheableDataCalc subclass
        :return: The data or None
        """
        assert calculator is CacheableDataCalc
        if self.cache_check(calculator):
            return self._cache[calculator.cid()]
        else:
            return None

    # da file
    def load(self, input_file):
        # TODO: implement loading from file (depends on _data type and cache)
        raise NotImplementedError("TODO")

    # su file
    def save(self, output_file):
        # TODO: implement saving to file (depends on _data type and cache)
        raise NotImplementedError("TODO")


# See README.md
class CacheableDataCalc(object):
    """ Static class that calculates cacheable data (like FFT etc.) """

    def __init__(self):
        raise TypeError('CacheableDataCalc is a static class')

    # metodo pubblico per ricavare i dati dalla cache o da _calculate_data(..)
    @classmethod
    def get(cls, data, params, use_cache=True):
        assert isinstance(data, DataSeries)
        if use_cache:
            if not data.cache_check(cls):
                data.cache_pre_calc_data(cls)
        else:
            return cls._calculate_data(data, params)
        return data.cache_get_data(cls)

    # metodo da sovrascrivere nelle sottoclassi
    @classmethod
    def _calculate_data(cls, data, params):
        raise NotImplementedError("Only on " + cls.__name__ + " sub-classes")

    # stringa usata come chiave nel dizionario cache
    @classmethod
    def cid(cls):
        """ Gets an identifier for the class
        :rtype : str
        """
        return cls.__name__ + "_cn"


# classe esempio per il calcolo dei dati intermedi
# TODO: copy-paste and/or edit this sample class
class FFTCalc(CacheableDataCalc):
    # basta sovrascrivere questo metodo

    @classmethod
    def _calculate_data(cls, data, params):
        """ Calculates the intermediate data
        :type data: DataSeries
        :param data: RRSeries object
        :param params: Params object
        :return: Data to cache
        """
        assert isinstance(data, DataSeries)
        # fittizio calcolo FFT
        # fft = data.series
        RR_interp, BT_interp=InterpolateRR(data.series, params)
        Finterp=params
        hw=np.hamming(len(RR_interp))

        frame=RR_interp*hw
        frame=frame-np.mean(frame)

        spec_tmp=np.absolute(np.fft.fft(frame))**2 # calcolo FFT
        spec = spec_tmp[0:(np.ceil(len(spec_tmp)/2))] # Only positive half of spectrum
        freqs = np.linspace(start=0,stop=Finterp/2,num=len(spec),endpoint=True) # creo vettore delle frequenze
        # ##
        return freqs, spec


class RRAnalysis(object):
    """ Static class containing methods for analyzing RR intervals data. """

    def __init__(self):
        raise NotImplementedError("RRAnalysis is a static class")

    @staticmethod
    def get_example_index(series):
        """ Example index method, returns the length
        :param series: DataSeries object to filter
        :return: DataSeries object filtered
        """
        assert type(series) is DataSeries
        return len(series.get_series)

    # TODO: add analysis scripts like in the example
    @staticmethod
    def TD_indexes(series):
        """ Returns TD indexes
        """
        assert type(series) is DataSeries
        RR=series.series
        RRmean=np.mean(RR)
        RRSTD= np.std(RR)

        RRDiffs=np.diff(RR)

        RRDiffs50 = [x for x in np.abs(RRDiffs) if x>50]
        pNN50=100.0*len(RRDiffs50)/len(RRDiffs)
        RRDiffs25 = [x for x in np.abs(RRDiffs) if x>25]
        pNN25=100.0*len(RRDiffs25)/len(RRDiffs)
        RRDiffs10 = [x for x in np.abs(RRDiffs) if x>10]
        pNN10=100.0*len(RRDiffs10)/len(RRDiffs)

        RMSSD = np.sqrt(sum(RRDiffs**2)/(len(RRDiffs)-1))
        SDSD = np.std(RRDiffs)

        labels= np.array(['RRmean', 'RRSTD', 'pNN50', 'pNN25', 'pNN10', 'RMSSD', 'SDSD'], dtype='S10')

        return [RRmean, RRSTD, pNN50, pNN25, pNN10, RMSSD,  SDSD], labels

    @staticmethod
    def POIN_indexes(series):
        """ Returns Poincare' indexes
        """
        assert type(series) is DataSeries
        RR=series.series
        # calculates Poincare' indexes
        xdata,ydata = RR[:-1], RR[1:]
        sd1 = np.std((xdata-ydata)/np.sqrt(2.0),ddof=1)
        sd2 = np.std((xdata+ydata)/np.sqrt(2.0),ddof=1)
        sd12 =sd1/sd2
        sEll=sd1*sd2*np.pi
        labels=['sd1', 'sd2', 'sd12', 'sEll']

        return [sd1, sd2,  sd12,  sEll],  labels

    @staticmethod
    def FD_indexes(series, Finterp):


        # freqs=np.arange(0, 2, 0.0001)
        #
        # # calculates AR coefficients
        # AR, P, k = spct.arburg(RR_interp*1000, 16) #burg
        #
        # # estimates PSD from AR coefficients
        # spec = spct.arma2psd(AR,  T=0.25, NFFT=2*len(freqs))
        freqs , spec = FFTCalc.get(series,Finterp, use_cache=True)

        # calculates power in different bands
        VLF=power(spec,freqs,0,0.04)
        LF=power(spec,freqs,0.04,0.15)
        HF=power(spec,freqs,0.15,0.4)
        Total=power(spec,freqs,0,2)
        LFHF = LF/HF
        nVLF=VLF/Total
        nLF=LF/Total
        nHF=HF/Total

        LFn=LF/(HF+LF)
        HFn=HF/(HF+LF)
        Power = [VLF, HF, LF]

        Power_Ratio= Power/sum(Power)
    #    Power_Ratio=spec/sum(spec) # uncomment to calculate Spectral Entropy using all frequencies
        Spectral_Entropy = 0
        lenPower=0 # tengo conto delle bande che ho utilizzato
        for i in xrange(0, len(Power_Ratio)):
            if Power_Ratio[i]>0: # potrei avere VLF=0
                Spectral_Entropy += Power_Ratio[i] * np.log(Power_Ratio[i])
                lenPower +=1
        Spectral_Entropy /= np.log(lenPower) #al posto di len(Power_Ratio) perche' magari non ho usato VLF

        labels= np.array(['VLF', 'LF', 'HF', 'Total', 'nVLF', 'nLF', 'nHF', 'LFn', 'HFn', 'LFHF', 'SpecEn'],  dtype='S10')

        return [VLF, LF, HF, Total, nVLF, nLF, nHF, LFn, HFn, LFHF, Spectral_Entropy], labels



class RRFilters(object):
    """ Static class containing methods for filtering RR intervals data. """

    def __init__(self):
        raise NotImplementedError("RRFilters is a static class")

    @staticmethod
    def example_filter(series):
        """ Example filter method, does nothing
        :param series: DataSeries object to filter
        :return: DataSeries object filtered
        """
        assert type(series) is DataSeries
        return series

    # TODO: add analysis scripts like in the example