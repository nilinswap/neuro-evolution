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
	rest_set=lis[0]#shared variable of array
	test_set=lis[1]#shared variable of array
	millis=lis[2]#lis

	n_par=10
	par_size=rest_set[0].get_value().shape[0]/10

	prmsdind=T.lscalar()
	valid_x_to_be=rest_set[0].get_value()[prmsdind*par_size:(prmsdind+1)*par_size,:]
	valid_y_to_be=rest_set[1].get_value()[prmsdind*par_size:(prmsdind+1)*par_size]
	train_x_to_be=np.concatenate((rest_set[0].get_value()[:(prmsdind)*par_size,:],rest_set[0].get_value()[(prmsdind+1)*par_size:,:]),axis=1)
	train_y_to_be=np.concatenate((rest_set[0].get_value()[:(prmsdind)*par_size,:],rest_set[0].get_value()[(prmsdind+1)*par_size:,:]),axis=1)
	
	"""classifier = LogisticRegression(input=x, n_in=28 * 28, n_out=10)

    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    cost = classifier.negative_log_likelihood(y)

    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to theta = (W,b)
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    # start-snippet-3
    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
	train_model=theao.function([prmsdind],outputs=,givens={x:train_x_to_be,y:train_y_to_be})




"""