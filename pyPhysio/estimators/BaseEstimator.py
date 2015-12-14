# coding=utf-8
__author__ = 'aleb'

from pyPhysio.BaseAlgorithm import Algorithm


class Estimator(Algorithm):

    @classmethod
    def algorithm(cls, data, params):
        """
        Placeholder for the subclasses
        @raise NotImplementedError: Ever
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
