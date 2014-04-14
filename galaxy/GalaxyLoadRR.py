__author__ = 'AleB'

from ParamExecClass import ParamExecClass
from pyHRV.Files import save_data_series


class GalaxyLoadRR(ParamExecClass):
    """
    kwargs['input'] ----> input file
    kwargs['output'] ---> output file
    kwargs['format'] ---> [ 'excel', 'csv' ]
                 default: 'csv'
    kwargs['column'] ---> column to load
                 default: PyHRVSettings.load_rr_column_name
    kwargs['sheet'] ---> excel's sheet's name or ordinal number
    """

    def execute(self):
        input_file = self._kwargs['input']
        output_file = self._kwargs['output']
        column = self._kwargs['column']

        save_data_series(self.load_column(), output_file)
