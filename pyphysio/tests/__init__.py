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

#from pytest import approx as approx

__author__ = 'aleb'


class TestData(object):
    _sing = None
    _path = _os.path.join(_os.path.dirname(__file__), "data")
    _file = "medical.txt.bz2"

    @classmethod
    def get_data(cls):
        if TestData._sing is None:
            TestData._sing = np.genfromtxt(_os.path.join(TestData._path, TestData._file), delimiter="\t")
        return TestData._sing

    # The following methods return an array to make it easier to test the Signal wrapping classes

    @classmethod
    def ecg(cls):
        return TestData.get_data()[:, 0]

    @classmethod
    def eda(cls):
        return TestData.get_data()[:, 1]

    @classmethod
    def bvp(cls):
        return TestData.get_data()[:, 2]

    @classmethod
    def resp(cls):
        return TestData.get_data()[:, 3]

#TODO: Add timeline