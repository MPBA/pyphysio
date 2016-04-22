from pyPhysio.galaxy.ParamExecClass import ParamExecClass

__author__ = 'AleB'

from pandas import DataFrame

from pyPhysio.segmentation import LinearTimeWindows
from pyPhysio.Files import *
from Extra.PyHRVSettings import MainSettings as Sett


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
        c = load_ds_from_csv_column(self._kwargs['input'])
        w = list(LinearTimeWindows(self._kwargs['step'], self._kwargs['width'], data=c))
        b, e = (map(lambda x: x.get_begin(), w), map(lambda x: x.get_end(), w))
        d = DataFrame(columns=['begin', 'end'])
        d['begin'] = b
        d['end'] = e
        d.to_csv(output_file, sep=Sett.load_csv_separator)