from pyPhysio.galaxy.ParamExecClass import ParamExecClass
from pyPhysio.Files import load_ds_from_csv_column, save_ds_to_csv
from pyPhysio.Filters import Filters

__author__ = 'AleB'


class GalaxyFilter(ParamExecClass):
    """
    T_RR_CSV -> T_RR_CSV
    kwargs['input'] ----> input file
    kwargs['output'] ---> output file
    """

    def execute(self):
        inp = self._kwargs['input']
        out = self._kwargs['output']
        save_ds_to_csv(Filters.filter_outliers(load_ds_from_csv_column(inp)), out)
