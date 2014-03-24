from pyHRV.DataSeries import DataSeries
from ParamExecClass import ParamExecClass

## DONE: (by the wrapper) if the input file is a .tar.gz: un-tar and load more files


class GalaxyLoadRR(ParamExecClass):
    """
    kwargs['input'] ----> input file
    kwargs['output'] ---> output file
    """
    def execute(self):
        input_file = self._kwargs['input']
        output_file = self._kwargs['output']

        rr_data = DataSeries()
        rr_data.load_from_csv(input_file)

        rr_data.to_csv(output_file, header=True)
