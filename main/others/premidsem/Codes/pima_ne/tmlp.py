
from __future__ import print_function

__docformat__ = 'restructedtext en'

import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit

import numpy 
from logreglayer import LogisticRegression
import theano
import theano.tensor as T
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

    def __init__(self, rng, input, n_in, n_hidden, n_out,weight_str=[]):
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
        self.n_in=n_in
        self.n_out=n_out
        self.n_hidden=n_hidden
        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function
        if len(weight_str):
            fir_weight = weight_str[1:(self.n_in+1)*self.n_hidden+1].reshape(self.n_in+1,self.n_hidden)
            fir_weightb=fir_weight[-1,:].reshape((self.n_hidden,))
            fir_weightrest=fir_weight[:-1,:]
            sec_weight = weight_str[(self.n_in+1)*self.n_hidden+1:].reshape((self.n_hidden+1),self.n_out)
            sec_weightb=sec_weight[-1,:].reshape((self.n_out,))
            sec_weightrest=sec_weight[:-1,:]
            fir_W_values = numpy.asarray(
                    fir_weightrest,
                    dtype=theano.config.floatX
                )
                

            self.fir_weight = theano.shared(value=fir_W_values, name='fir_W', borrow=True)

            fir_b_values =numpy.asarray(
                    fir_weightb,
                    dtype=theano.config.floatX
                )
            self.fir_b = theano.shared(value=fir_b_values, name='fir_b', borrow=True)
            
            sec_W_values = numpy.asarray(
                    sec_weightrest,
                    dtype=theano.config.floatX
                )
            self.sec_weight = theano.shared(value=sec_W_values, name='sec_W', borrow=True)
            sec_b_values =numpy.asarray(
                    sec_weightb,
                    dtype=theano.config.floatX
                )
            self.sec_b = theano.shared(value=sec_b_values, name='sec_b', borrow=True)
            print(self.fir_weight,self.fir_b)
        self.hiddenLayer = hiddenlayer.HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh,
            W=self.fir_weight,
            b=self.fir_b
            )
        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out,
            W=self.sec_weight,
            b=self.sec_b
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
        self.mean_square_error=(
            self.logRegressionLayer.mean_square_error
        )
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params
        # end-snippet-3

        # keep track of model input
        self.input = input