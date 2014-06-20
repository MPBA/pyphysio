from galaxy import ParamExecClass

__author__ = 'AleB'

from pandas import DataFrame

from pyHRV.windowing import LinearTimeWinGen
from pyHRV.Files import *
from pyHRV.PyHRVSettings import PyHRVDefaultSettings as Sett


class GalaxyLinearTimeWindows(ParamExecClass):
    """
    T_FILE_LABELS -> T_WIN_COLLECTION_CSV
    kwargs['input'] ----> input file
    kwargs['output'] ---> output file
    kwargs['step'] ---> step between two windows
    kwargs['width'] ---> width of the window
    """

    def execute(self):
        output_file = self._kwargs['output']
        c = load_rr(self._kwargs['input'])
        w = list(LinearTimeWinGen(self._kwargs['step'], self._kwargs['width'], data=c))
        b, e = (map(lambda x: x.begin, w), map(lambda x: x.end, w))
        d = DataFrame(columns=['begin', 'end'])
        d['begin'] = b
        d['end'] = e
        d.to_csv(output_file, sep=Sett.load_csv_separator)
