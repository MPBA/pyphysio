import os
import sys

sys.path.insert(0, os.path.abspath('..'))

# Used by test cases so:
# noinspection PyUnresolvedReferences,PyPep8Naming
import pyPhysio as ph


class Asset(object):
    asset = "assets/"
    F18 = asset + "F18.txt"
    BVP = asset + "BVP.csv"
    GSR = asset + "GSR.csv"
