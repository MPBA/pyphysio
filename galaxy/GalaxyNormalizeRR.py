__author__ = 'AleB'

from ParamExecClass import ParamExecClass
from pyHRV.Files import load_rr, save_data_series
from pyHRV.Filters import RRFilters


class GalaxyNormalizeRR(ParamExecClass):
    """
    T_RR_CSV -> T_RR_CSV
    kwargs['input'] ----> input file
    kwargs['output'] ---> output file
    kwargs['column'] ---> column to load
                 default(None): PyHRVSettings.load_rr_column_name
    kwargs['norm_mode'] ---> normalization mode [ 'mean', 'mean_sd', 'min', 'max_min' ]
    """

    def execute(self):
        inp = self._kwargs['input']
        out = self._kwargs['output']
        save_data_series(
            getattr(RRFilters, "normalize_" + self._kwargs['norm_mode'])(load_rr(inp)), out)
