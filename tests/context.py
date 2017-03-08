# coding=utf-8
import numpy as np
import os


class Assets(object):
    _sing = None
    _path = "assets/"
    _file_c = "medical.txt.bz2"
    _file_u = "medical.txt"

    @classmethod
    def get_data(cls):
        if Assets._sing is None:
            if not os.path.isfile(Assets._path + Assets._file_c):
                Assets._path = "../" + Assets._path
            if not os.path.isfile(Assets._path + Assets._file_u):
                os.system("bzcat %s%s > %s%s" % (Assets._path, Assets._file_c, Assets._path, Assets._file_u))
            Assets._sing = np.genfromtxt(Assets._path + Assets._file_u, delimiter="\t")
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
