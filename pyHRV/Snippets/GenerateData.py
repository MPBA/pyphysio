__author__ = 'ale'

import numpy
import pandas

# We create as example a random database in the file "RD.txt"
ls = numpy.random.randint(50, 80, 100)
ln = ["Red", "Green", "Blue", "Relaxed", "Noise"]
ll = []
# noinspection PyTypeChecker
for s in ls:
    # noinspection PyArgumentList
    l = ln[int(numpy.random.rand() * len(ln))]
    for i in xrange(int(s)):
        ll.append(l)
rr = numpy.random.randint(500, 1500, len(ll))
pandas.DataFrame({"IBI": rr, "label": ll}).to_csv("RD.txt", sep="\t", index=False, header=True)
