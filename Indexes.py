# coding=utf-8
__author__ = 'AleB'


from utility import interpolate_rr


class DataAnalysis(object):
    pass


class Index(object):
    def __init__(self, data=None):
        self._value = None
        self._data = data

    @property
    def calculated(self):
        """
        Returns weather the index is alredy calculated and up-to-date
        @return: Boolean
        """
        return not (self._value is None)

    @property
    def value(self):
        return self._value

    def update(self, data):
        self._data = data
        self._value = None


class TDIndex(Index):
    def __init__(self, data=None):
        super(TDIndex, self).__init__(data)


class FDIndex(Index):
    def __init__(self, data=None):
        super(FDIndex, self).__init__(data)

    # TODO: mi sembra si possa migliorare dato che c'è una possibile perdita
    # TODO: di informazioni (conversione) DataSeries ->-> np
    def _interpolate(self, to_freq):
        """
        Privata. Interpola quando chiamata dalle sottoclassi
        @param to_freq:
        @return:
        """
        rr_interp, bt_interp = interpolate_rr(self._data, to_freq)
        self._data = rr_interp

    def _estimate_psd(self, fsamp, method):
        # TODO: estimate PSD (non trovo il codice)
        pass

