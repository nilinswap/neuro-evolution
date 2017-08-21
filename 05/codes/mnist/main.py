from __future__ import print_function

__docformat__ = 'restructedtext en'

import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T
import load_data
def main():

	lis=load_data.load_data(dataset)
	rest_set=lis[0]
	test_set=lis[1]
	millis=lis[2]

	