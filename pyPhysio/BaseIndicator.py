# coding=utf-8
from BaseAlgorithm import Algorithm
from abc import ABCMeta as _ABCMeta
__author__ = 'AleB'


class Indicator(Algorithm):
    __metaclass__ = _ABCMeta

    @classmethod
    def compute_on(cls, state):
        """
        For on-line mode.
        @param state: Support values
        @raise NotImplementedError: Ever here.
        """
        raise TypeError(cls.__name__ + " is not implemented as an on-line feature.")

    @classmethod
    def required_sv(cls):
        """
        Returns the list of the support values that the on-line version of the algorithm requires.
        @rtype: list
        """
        return []
