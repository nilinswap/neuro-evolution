
from __future__ import print_function

__docformat__ = 'restructedtext en'

import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit

import numpy 
from tf_logreglayer import LogisticRegression

import tensorflow as tf
import tf_hiddenlayer
class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softmax layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, input, n_in, n_out, n_hidden, rng):
        """Initialize the parameters for the multilayer perceptron
        """
        self.n_hidden=n_hidden
        self.n_in=n_in
        self.n_out=n_out
        self.n_hid=n_hidden

        self.hiddenLayer = tf_hiddenlayer.HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=tf.tanh
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
        self.L1 = tf.add(
            tf.reduce_sum(tf.abs(self.hiddenLayer.W))
            ,tf.reduce_sum(tf.abs(self.logRegressionLayer.W))
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = tf.add(
            tf.reduce_sum(tf.pow(self.hiddenLayer.W,2)) ,
            tf.reduce_sum(tf.pow(self.logRegressionLayer.W ,2))
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
    def get_weights(self):
        
            
        w2=self.logRegressionLayer.W.eval()
        b2=self.logRegressionLayer.b.eval()
        w1=self.hiddenLayer.W.eval()
        b1=self.hiddenLayer.b.eval()

        tup=(w1,b1,w2,b2)
        return tup
    def set_weights(self,w1,b1,w2,b2):
        p=self.logRegressionLayer.W.assign(w2)
        q=self.logRegressionLayer.b.assign(b2)
        r=self.hiddenLayer.W.assign(w1)
        s=self.hiddenLayer.b.assign(b1)
        with tf.Session() as sess:
            sess.run([p,q,r,s])
    def set_weights_from_chromosome(self,weightstr):
        new_weightstr=weightstr[1:]
        n_hid=int(weightstr[0])

        w1=new_weightstr[:(self.n_in*self.n_hid)].reshape((self.n_in,self.n_hid))
        b1=new_weightstr[(self.n_in*self.n_hid):(self.n_in*self.n_hid)+self.n_hid].reshape((self.n_hid,))
        w2=new_weightstr[(self.n_in*self.n_hid)+self.n_hid:(self.n_in*self.n_hid)+self.n_hid+(self.n_hid*self.n_out)].reshape((self.n_hid,self.n_out))
        b2=new_weightstr[(self.n_in*self.n_hid)+self.n_hid+(self.n_hid*self.n_out):].reshape((self.n_out,))
        self.set_weights(w1,b1,w2,b2)
    def turn_weights_into_chromosome(self):
        w1,b1,w2,b2=self.get_weights()
        w1=list(w1.reshape(((self.n_in*self.n_hid),)))
        
        b1=list(b1)
        w2=list(w2.reshape(((self.n_hid*self.n_out),)))
        b2=list(b2)

        lis=[float(self.n_hid)]
        lis=lis+w1+b1+w2+b2
        np=numpy
        return np.array(lis)
