from pyHRV.DataSeries import DataSeries
from ParamExecClass import ParamExecClass
from pyHRV.Files import load_rr_data_series, save_rr_data_series

## DONE: (by the wrapper) if the input file is a .tar.gz: un-tar and load more files


class GalaxyLoadRR(ParamExecClass):
    """
    kwargs['input'] ----> input file
    kwargs['output'] ---> output file
    kwargs['column'] ---> column to load
                 default: PyHRVSettings.load_rr_column_name
    """
    def execute(self):
        input_file = self._kwargs['input']
        output_file = self._kwargs['output']
        column = self._kwargs['column']

        save_rr_data_series(load_rr_data_series(input_file, column=column), output_file)
