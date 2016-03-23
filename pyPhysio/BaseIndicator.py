# coding=utf-8
from BaseAlgorithm import Algorithm
__author__ = 'AleB'


class Indicator(Algorithm):
    """
    Defines a class of algorithms that extracts a single time tuple, only one time-value pair, that has specific data as
    value.
    """

    @classmethod
    def is_nature_supported(cls, data):
        return True

    @classmethod
    def algorithm(cls, data, params):
        """
        Placeholder for the subclasses
        @raise NotImplementedError: Ever
        :param params:
        :param data:
        """
        raise NotImplementedError(cls.__name__ + " is a Feature but it is not implemented.")

    @classmethod
    def get_used_params(cls):
        """
        Default empty list of parameters.
        @return An empty list if not overridden.
        """
        return []

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
