# coding=utf-8
import os as _os
try:
    import pyphysio as ph
except ImportError:
    from sys import path
    path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
    import pyphysio as ph
import numpy as np
# noinspection PyUnresolvedReferences
from pytest import approx as approx


class Assets(object):
    _sing = None
    _path = "assets/"
    _file_c = "medical.txt.bz2"
    _file_u = "medical.txt"

    @classmethod
    def get_data(cls):
        if Assets._sing is None:
            if not _os.path.isfile(Assets._path + Assets._file_c):
                Assets._path = "../" + Assets._path
            if not _os.path.isfile(Assets._path + Assets._file_u):
                _os.system("bzcat %s%s > %s%s" % (Assets._path, Assets._file_c, Assets._path, Assets._file_u))
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
