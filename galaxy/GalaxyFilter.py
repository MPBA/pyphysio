__author__ = 'AleB'

from ParamExecClass import ParamExecClass
from pyHRV.Files import load_rr, save_data_series
from pyHRV.Filters import RRFilters


class GalaxyFilter(ParamExecClass):
    """
    T_RR_CSV -> T_RR_CSV
    kwargs['input'] ----> input file
    kwargs['output'] ---> output file
    kwargs['column'] ---> column to load
                 default(None): PyHRVSettings.load_rr_column_name
    """

    def execute(self):
        inp = self._kwargs['input']
        out = self._kwargs['output']
        save_data_series(RRFilters.filter_outliers(load_rr(inp, self._kwargs['column'])), out)
