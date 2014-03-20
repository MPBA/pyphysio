from PyHRV.DataSeries import DataSeries

## DONE: (by the wrapper) if inputfile is a .tar.gz: untar and load more files


class GalaxyLoadRR(object):
    @staticmethod
    def execute(**kwargs):
        input_file = kwargs['input']
        output_file = kwargs['output']
        separator = kwargs['sep']
        column_name = kwargs['colname']

        rr_data=DataSeries()
        rr_data.load_from_csv(input_file, sep=separator, colname=column_name)

        rr_data.to_csv(output_file, header=True)
