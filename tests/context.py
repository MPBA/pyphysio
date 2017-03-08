# coding=utf-8
import numpy as np


class Assets(object):
    _sing = None

    @classmethod
    def get_data(cls):
        if Assets._sing is None:
            try:
                Assets._sing = np.genfromtxt("assets/medical.txt", delimiter="\t")
            except IOError:
                Assets._sing = np.genfromtxt("../assets/medical.txt", delimiter="\t")
        return Assets._sing

    # The following methods return an array to make it easier to test the Signal wrapping classes

    @classmethod
    def ecg(cls):
        return Assets.get_data()[:, 0]

    @classmethod
    def eda(cls):
        return Assets.get_data()[:, 1]

    @classmethod
    def bvp(cls):
        return Assets.get_data()[:, 2]

    @classmethod
    def resp(cls):
        return Assets.get_data()[:, 3]

# Used by test cases so:
# noinspection PyUnresolvedReferences,PyPep8Naming
import pyphysio as ph
