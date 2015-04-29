from pyPhysio.galaxy.ParamExecClass import ParamExecClass

__author__ = 'AleB'

from pyPhysio.Files import save_ds_to_csv, load_pd_from_excel_column


class GalaxyLoadRRExcel(ParamExecClass):
    """
    FILE_EXCEL -> T_RR_CSV
    kwargs['input'] ----> input file
    kwargs['output'] ---> output file
    kwargs['column'] ---> column to load
                 default: PyHRVSettings.load_rr_column_name
    kwargs['sheet'] ---> excel's sheet's name or ordinal number
    """

    def execute(self):
        out = self._kwargs['output']

        save_ds_to_csv(load_pd_from_excel_column(self._kwargs['input'], self._kwargs['column'],
                                                 None, self._kwargs['sheet']), out)
