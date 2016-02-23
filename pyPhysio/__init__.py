# coding=utf-8
from __future__ import division

__author__ = "AleB"

from features import *
from filters import *
from segmentation import *
from BaseSegmentation import Segment
from WindowsIterator import WindowsIterator


class PhUI(object):
    @staticmethod
    def a(condition, message):
        if not condition:
            raise ValueError(message)

    @staticmethod
    def i(mex):
        PhUI.p(mex, '', 35)

    @staticmethod
    def o(mex):
        PhUI.p(mex, '', 31)

    @staticmethod
    def w(mex):
        PhUI.p(mex, 'Warning: ', 33)

    @staticmethod
    def p(mex, lev, col):
        print(">%s\x1b[%dm%s\x1b[39m" % (lev, col, mex))
