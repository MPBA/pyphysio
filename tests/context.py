# coding=utf-8
import numpy as np


class Assets(object):
    _sing = None

    @classmethod
    def get_data(cls):
        if Assets._sing is None:
            try:
                Assets._sing = np.genfromtxt("assets/medical.txt", delimiter="\t", max_rows=10000)
            except IOError:
                Assets._sing = np.genfromtxt("../assets/medical.txt", delimiter="\t", max_rows=10000)
        return Assets._sing

    @classmethod
    def ecg(cls):
        return Assets.get_data()[:, 0]

    @classmethod
    def gsr(cls):
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
