#!/usr/bin/python3
# coding=utf-8
from __future__ import print_function
from sys import argv

if len(argv) < 2:
    print("Usage:", argv[0], "Superclass [param1 param2 ...]")
else:
    sign = ""
    call = ""
    for i in argv[2:]:
        if i[0] == "?":
            sign += i[1:] + "=None, "
            call += i[1:] + "=" + i[1:] + ", "
        else:
            sign += i + ", "
            call += i + "=" + i + ", "
    print("    def __init__(self, ",
          sign,
          "**kwargs):\n        ",
          argv[1],
          ".__init__(self, ",
          call,
          "**kwargs)",
          sep='')
