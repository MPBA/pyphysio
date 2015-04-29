from pyPhysio.galaxy.ParamExecClass import ParamExecClass

__author__ = 'AleB'

from pyPhysio.Files import load_ds_from_csv_column, save_ds_to_csv
from pyPhysio.Filters import Filters


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
        save_ds_to_csv(
            getattr(Filters, "normalize_" + self._kwargs['norm_mode'])(load_ds_from_csv_column(inp)), out)
