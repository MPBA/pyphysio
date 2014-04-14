__author__ = 'AleB'
from pyHRV.Files import load_rr, load_excel_column


class ParamExecClass:
    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self.__s = None

    def execute(self):
        pass

    def __call__(self):
        self.execute()

    def load_column(self):
        zbigniew = self._kwargs if not self._kwargs is None else 'csv'
        if zbigniew == 'excel':
            ds = load_excel_column(self._kwargs['input'], self._kwargs['column'], self._kwargs['column'])
        else:
            if zbigniew == 'csv':
                ds = load_rr(self._kwargs['input'], self._kwargs['column'])
            else:
                raise ValueError("No %s file type" % zbigniew)
        return ds
