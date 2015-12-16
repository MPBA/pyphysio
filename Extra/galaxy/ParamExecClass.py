__author__ = 'AleB'


class ParamExecClass:
    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self.__s = None

    def execute(self):
        pass

    def __call__(self):
        self.execute()
