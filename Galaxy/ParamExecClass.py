

class ParamExecClass:
    def __init__(self, input_path, output_path, **kwargs):
        self._input_path = input_path
        self._output_path = output_path
        self._kwargs = kwargs

    def execute(self):
        pass

    def __call__(self):
        self.execute()
