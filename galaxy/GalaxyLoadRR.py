__author__ = 'AleB'

from ParamExecClass import ParamExecClass
from pyHRV.Files import *


class GalaxyLoadRR(ParamExecClass):
    """
    FILE_CSV -> T_RR_CSV
    kwargs['input'] ----> input file
    kwargs['output'] ---> output file
    kwargs['data_type'] ---> [ 'ecg', 'bvp', 'rr' ]
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
            ds = load_rr(inp)
        else:
            raise ValueError("data_type is " + d)
        save_data_series(ds, out)
