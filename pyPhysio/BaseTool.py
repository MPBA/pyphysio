# coding=utf-8
from pyPhysio.BaseAlgorithm import Algorithm

__author__ = 'AleB'


class Tool(Algorithm):
    """
    Algorithms which take as input a signal and return one or more np.array
    """
    pass

    @classmethod
    def get_signal_type(cls):
        return None

    @classmethod
    def is_compatible(cls, signal):
        return cls.get_signal_type() is None or signal.get_signal_nature() in cls.get_signal_type()
