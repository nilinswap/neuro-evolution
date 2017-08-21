import numpy 

import theano
import theano.tensor as T

import load_data

def yieldnewset(dataset):

	lis=load_data.load_data(dataset)
	rest_set=lis[0]
	test_set=lis[1]
	millis=lis[2]

	