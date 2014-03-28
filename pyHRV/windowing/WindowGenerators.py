__author__ = 'AleB'
__all__ = ['LinearWinGen']
from WindowsBase import *


class LinearWinGen(WindowsGenerator):
    def __init__(self, data, begin, step, width, end):
        super(LinearWinGen, self).__init__(data)
        self._begin = begin
        self._end = end
        self._step = step
        self._width = width
        self._pos = begin

    def step_windowing(self):
        w = (self._pos, self._pos + self._width)
        self._pos += self._step
        return Window(*w)
