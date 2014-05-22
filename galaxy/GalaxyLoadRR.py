__author__ = 'AleB'

from ParamExecClass import ParamExecClass
from pyHRV.Files import *


class GalaxyLoadRR(ParamExecClass):
    """
    kwargs['input'] ----> input file
    kwargs['output'] ---> output file
    kwargs['data_type'] ---> [ 'ecg', 'bvp', 'rr' ]
    kwargs['column'] ---> column to load
                 default(None): PyHRVSettings.load_rr_column_name
    """

    def execute(self):
        inp = self._kwargs['input']
        out = self._kwargs['output']
        d = self._kwargs['data_type']
        if d == 'ecg':
            ds = load_rr_from_ecg(inp)
        elif d == 'bvp':
            ds = load_rr_from_bvp(inp)
        elif d == 'rr':
            ds = load_rr(inp, self._kwargs['column'])
        else:
            raise ValueError("data_type is " + d)
        save_data_series(ds, out)
