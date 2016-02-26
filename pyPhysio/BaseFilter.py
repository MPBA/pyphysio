# coding=utf-8
from BaseAlgorithm import Algorithm
__author__ = 'AleB'


class Filter(Algorithm):

    @classmethod
    def algorithm(cls, data, params):
        """
        Placeholder for the subclasses
        @raise NotImplementedError: Ever
        :param params:
        :param data:
        """
        raise NotImplementedError(cls.__name__ + " is not implemented.")

    @classmethod
    def get_used_params(cls):
        """
        The default return if the method is not implemented
        :return: Empty list
        :rtype: list
        """
        return []
