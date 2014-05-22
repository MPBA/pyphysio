__author__ = 'AleB'

from pandas import Series

from ParamExecClass import ParamExecClass
from pyHRV.windowing import NamedWinGen, CollectionWinGen, Window
from pyHRV.Files import *


class GalaxyLoadWindows(ParamExecClass):
    """
    kwargs['input'] ----> input file
    kwargs['output'] ---> output file
    kwargs['format'] ---> one in [ 'excel', 'csv' ]
                 default: 'csv'
    kwargs['column'] ---> column of the windows to load
    kwargs['sheet'] ---> excel's sheet's name or ordinal number
                 default: None if format != 'excel'
    kwargs['windows_type'] ---> windows type to load in
                                [ 'labels_sequences_linear', 'labels_sequences', 'begin_values' ]
    """

    def execute(self):
        output_file = self._kwargs['output']
        if self._kwargs['excel']:
            c = load_excel_column(self._kwargs['input'], self._kwargs['column'], None, self._kwargs['sheet'])
        else:
            c = load_rr(self._kwargs['input'], self._kwargs['column'])

        if self._kwargs['windows_type'] == 'labels_sequences':
            w = NamedWinGen(None, c)
        else:
            if self._kwargs['windows_type'] == 'begin_values':
                w = map(self.__class__.map_end_window, c)[1:]
                w = CollectionWinGen(None, w)
            else:
                raise NotImplemented("Not implemented windowing mode: %s" % self._kwargs['windows_type'])
        Series(w).save(output_file)

    def map_end_window(self, end):
        if self.__s is None:
            self.__s = end
        else:
            return Window(self.__s, end)
