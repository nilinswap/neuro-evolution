from __future__ import print_function

__docformat__ = 'restructedtext en'

import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit
from theano.ifelse import ifelse
import numpy as np

import theano
import theano.tensor as T
import load_data
from logreglayer import LogisticRegression
import hiddenlayer
class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softmax layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """

        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function
        self.hiddenLayer = hiddenlayer.HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh
        )

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out
        )
        # end-snippet-2 start-snippet-3
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params
        # end-snippet-3

        # keep track of model input
        self.input = input

def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
             dataset='cards.data', n_par=10, n_hidden=100,freq_par=0.9):
	
	
	
	lis=load_data.load_data(dataset)
	
	rest_set=lis[0]#tuple of two shared variable of array
	test_set=lis[1]#tuple of shared variable of array

	millis=lis[2]#lis

	
	par_size=int(rest_set[0].get_value().shape[0]/10)
	#print(rest_set[0].get_value().shape[0])
	#
	prmsdind=T.lscalar()
	
	
	valid_x_to_be=rest_set[0][prmsdind*par_size:(prmsdind+1)*par_size,:]
	
	

	valid_y_to_be=rest_set[1][prmsdind*par_size:(prmsdind+1)*par_size]
	train_x_to_be=T.concatenate((rest_set[0][:(prmsdind)*par_size,:],rest_set[0][(prmsdind+1)*par_size:,:]),axis=0)
	train_y_to_be=T.concatenate((rest_set[1][:(prmsdind)*par_size],rest_set[1][(prmsdind+1)*par_size:]))
	#train_fun=theano.function(,givens={x:train_x_to_be,y:train_y_to_be})
	
	fun=theano.function([prmsdind],valid_y_to_be)
	

	#cool thing starts from here ->
    ######################
    # BUILD ACTUAL MODEL #
    ######################
	print('...building the model')

	# allocate symbolic variables for the data
	#index = T.lscalar()  # index to a [mini]batch
	x = T.matrix('x')  # the data is presented as rasterized images
	y = T.ivector('y')  # the labels are presented as 1D vector of
	                    # [int] labels

	rng = np.random.RandomState(1234)



	# construct the MLP class
	classifier = MLP(
	    rng=rng,
	    input=x,
	    n_in=15,
	    n_hidden=n_hidden,
	    n_out=2
	)

	# start-snippet-4
	# the cost we minimize during training is the negative log likelihood of
	# the model plus the regularization terms (L1 and L2); cost is expressed
	# here symbolically
	cost = (
	    classifier.negative_log_likelihood(y)
	    + L1_reg * classifier.L1
	    + L2_reg * classifier.L2_sqr
	)
	# end-snippet-4

	# compiling a Theano function that computes the mistakes that are made
	# by the model on a minibatch
	test_x_to_be=test_set[0]
	test_y_to_be=test_set[1]
	test_model = theano.function(
	    inputs=[prmsdind],
	    outputs=classifier.errors(y),
	    givens={
	    	x:test_x_to_be,
	    	y:test_y_to_be
	    },
	    on_unused_input='ignore'
	)

	validate_model = theano.function(
	    inputs=[prmsdind],
	    outputs=classifier.errors(y),
	    givens={
	        x: valid_x_to_be,
	        y: valid_y_to_be
	    }
	)

	# start-snippet-5
	# compute the gradient of cost with respect to theta (sorted in params)
	# the resulting gradients will be stored in a list gparams
	gparams = [T.grad(cost, param) for param in classifier.params]

	# specify how to update the parameters of the model as a list of
	# (variable, update expression) pairs

	# given two lists of the same length, A = [a1, a2, a3, a4] and
	# B = [b1, b2, b3, b4], zip generates a list C of same size, where each
	# element is a pair formed from the two lists :
	#    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
	updates = [
	    (param, param - learning_rate * gparam)
	    for param, gparam in zip(classifier.params, gparams)
	]

	# compiling a Theano function `train_model` that returns the cost, but
	# in the same time updates the parameter of the model based on the rules
	# defined in `updates`
	train_model = theano.function(
	    inputs=[prmsdind],
	    outputs=cost,
	    updates=updates,
	    givens={
	        x: train_x_to_be,
	        y: train_y_to_be 
	        }
	)

	#setting condition to increase frequency of validation with epochs done as first thing inside 'i in n_epochs' loop
	epo=T.lscalar()

	epostep=theano.shared(100)
	eqh=ifelse(T.eq(epo%epostep,0),1,0)
	ie=ifelse(T.lt(1,epostep),eqh,1)
	booly=theano.function([epo],ie)

	#here it ends

	# end-snippet-5

	###############
	# TRAIN MODEL #
	###############
	minavg=np.inf
	lasttolastavg=10002
	lastavg=10001
	presentavg=10000
	flag=0
	for i in range(n_epochs):
		if booly(i):
			epostep.set_value(int(epostep.get_value()*freq_par))
			
			avrg=0
		for ind in range(n_par):
			trainerr=train_model(ind)
			if booly(i):
				validerr=validate_model(ind)
				avrg=(avrg*(ind)+validerr)/(ind+1)
		if booly(i):
	 		if minavg>avrg:
	 			minavg=avrg
	 		lasttolastavg=lastavg
	 		lastavg=presentavg
	 		presentavg=avrg
	 		print(presentavg)
	 		if not ((( lastavg- presentavg) > 0.001) or ((lasttolastavg - lastavg)>0.001)):
	 			flag=1
		if flag:
			break

	if not flag:
		print("increase epochs maybe")
	print("achieved min err as", presentavg)
	print("test error is ",test_model(0))

def main():
 	test_mlp()
if __name__ == '__main__':
    main()






