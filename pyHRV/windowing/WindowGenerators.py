__author__ = 'AleB'
__all__ = ['LinearWinGen']
from WindowsBase import WindowsGenerator, Window


class LinearWinGen(WindowsGenerator):
    """Generates a linear set of windows (b+i*s, b+i*s+w)
    """

    def __init__(self, data, begin, step, width, end):
        super(LinearWinGen, self).__init__(data)
        self._begin = begin
        self._end = end
        self._step = step
        self._width = width
        self._pos = begin

    def step_windowing(self):
        b, e = (self._pos, self._pos + self._width)
        if e > self._end:
            raise StopIteration
        else:
            self._pos += self._step
            return Window(b, e)
