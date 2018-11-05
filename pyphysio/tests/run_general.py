# coding=utf-8
from __future__ import division

from . import ph
import unittest

__author__ = 'aleb'


def test_annotate_ecg():
    ecg = ph.EvenlySignal(ph.TestData.ecg(), 2048)
    ibi = ph.BeatFromECG()(ecg)

    print(ph.Annotate()(ecg.segment_time(0, 20), []))


if __name__ == '__main__':
    unittest.main()
