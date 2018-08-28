#!/usr/bin/env python
# noispection PyPep8Naming

'''
 This is template only, but NOT WORKING example. 
'''

import os
import numpy as np
import multiprocessing
import itertools
import fnmatch
import matplotlib.pyplot as plt
import cPickle as pickle

import tfmicro.threadgen


# noinspection PyUnresolvedReferences
class Data(object):
    def __init__(self, c):
        self.c = c
        self.prepare()
        self.unit_tests()

    # prepare your data here
    def prepare(self):
        c = self.c
        self.batch_size = c['model.batch_size']

    # generator
    @tfmicro.threadgen.threadsafe_generator
    def generator(self, mode):
        c = self.c
        yield None

    # unit testing
    def unit_tests(self, debug=False):
       return False


if __name__ == '__main__':
    import json
    d = Data(json.load(open('config.json')))
    d.unit_tests(debug=True)






