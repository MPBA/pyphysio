__author__ = 'AleB'

from ParamExecClass import ParamExecClass
from pyHRV.Files import load_data_series, save_data_series
from pyHRV.Filters import RRFilters


class GalaxyNormalizeRR(ParamExecClass):
    """
    T_RR_CSV -> T_RR_CSV
    kwargs['input'] ----> input file
    kwargs['output'] ---> output file
    kwargs['norm_mode'] ---> normalization mode [ 'mean', 'mean_sd', 'min', 'max_min' ]
    """

    def execute(self):
        inp = self._kwargs['input']
        out = self._kwargs['output']
        save_data_series(
            getattr(RRFilters, "normalize_" + self._kwargs['norm_mode'])(load_data_series(inp)), out)
