

from __future__ import print_function

__docformat__ = 'restructedtext en'

import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit
from functools import reduce
import numpy as np

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
def func(last,current):
        return [last[0]+1,current]

class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
       
        """self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        """
        self.W=tf.Variable(initial_value=np.random.random((n_in, n_out)),  name='W', dtype='float64')
        # initialize the biases b as a vector of n_out 0s
        self.b = tf.Variable(initial_value=np.random.random(( n_out,)),  name='b', dtype='float64')

        if int(self.b.shape[0])!=1:

            self.p_y_given_x = tf.nn.softmax(tf.add(tf.matmul(input, self.W),self.b))
        else:
            self.p_y_given_x = tf.nn.sigmoid(tf.add(tf.matmul(input, self.W),self.b))
        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        half=tf.constant(0.5,dtype=self.p_y_given_x.dtype)
        if int(self.b.shape[0])!=1:
            self.y_pred = tf.argmax(self.p_y_given_x, axis=1)
        else:
            half=tf.constant(0.5,dtype=self.p_y_given_x.dtype)
            dadum=tf.constant(0.5,dtype=self.p_y_given_x.dtype)
            q=tf.scan(lambda last,current: current[0],elems=self.p_y_given_x,initializer=dadum)
            s=tf.scan(lambda y,x: tf.greater_equal(x,half),elems=q,initializer=False)
            print("herehrerhehrehrehrehrhe",s)
            #print("hi",s)
            self.y_pred=tf.cast(s,dtype=tf.int32)

        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

    
        

    def negative_log_likelihood(self, y):
        if int(self.b.shape[0])!=1: 
           dum=tf.constant(0.5,dtype=tf.float64) #dum for dummy
           dadum=tf.constant(-1,dtype=tf.int32)# dum-dadum-dadum mast h
           q=tf.scan(fn=func,elems=y,initializer=[dadum,dadum])
           z=tf.transpose(tf.stack([q[0],q[1]]))
           #print("hello---------------------------")
           w=tf.scan(lambda last,current: tf.log(self.p_y_given_x[current[0]][current[1]]),elems=z,initializer=dum)
           #print(-tf.reduce_mean(w))
           return -tf.reduce_mean(w)
        else:

            dum=tf.constant(0.5,dtype=tf.float64)
            minusone=tf.constant(-1,dtype=tf.int32)
            one=tf.constant(1,dtype=y.dtype)
            r=tf.scan(lambda last,current:last+1,elems=y,initializer=minusone)

            w=tf.scan(lambda last,current: tf.add(tf.multiply(tf.cast(y[current],dtype=self.p_y_given_x.dtype),tf.log(self.p_y_given_x[current][0])),tf.multiply(tf.cast(tf.add(one,-y[current]),dtype=self.p_y_given_x.dtype),tf.log(tf.add(tf.cast(one,dtype=self.p_y_given_x.dtype),-self.p_y_given_x[current][0])))),elems=r,initializer=dum)
            z=-tf.reduce_mean(w)
            return z



    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        """if len(y.shape) != len(self.y_pred.shape):
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        """
        # check if y is of the correct datatype

        if y.dtype:
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            r=tf.scan(lambda last,current:last+1,elems=y,initializer=-1)
            qn=tf.scan(lambda last,current: tf.not_equal(tf.cast(self.y_pred[current],dtype=tf.int32),y[current]),elems=r,initializer=False)
            q=tf.cast(qn,dtype=tf.int32)
            
            #r=tf.scan((lambda last,current: current[1]),q)
            return tf.reduce_mean(tf.cast(q,dtype=tf.float64))
        else:
            raise NotImplementedError()








def test_main():

    x=tf.placeholder('float64',shape=[None,5])
    n_in=5
    n_out=2
    size=4
    tar=np.random.randint(0,2,(size,))
    xar=np.random.random((size,n_in))
    obj=LogisticRegression(x,n_in,n_out)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run([obj.negative_log_likelihood(tar),obj.p_y_given_x],feed_dict={x:xar}))
        

if __name__=='__main__':
    test_main()