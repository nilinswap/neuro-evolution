

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
        self.W=tf.Variable(initial_value=np.zeros((n_in, n_out)),  name='W', dtype='float64')
        # initialize the biases b as a vector of n_out 0s
        self.b = tf.Variable(initial_value=np.zeros(( n_out,)),  name='b', dtype='float64')

        
        self.p_y_given_x = tf.nn.softmax(tf.add(tf.matmul(input, self.W),self.b))

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = tf.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

    

    def negative_log_likelihood(self, y):
       
       dum=tf.constant(0.5,dtype=tf.float64) #dum for dummy
       dadum=tf.constant(-1,dtype=tf.int32)
       q=tf.scan(fn=func,elems=y,initializer=[dadum,dadum])
       z=tf.transpose(tf.stack([q[0],q[1]]))
       print("hello---------------------------")
       w=tf.scan(fn=(lambda last,current: tf.log(self.p_y_given_x[current[0]][current[1]])),elems=z,initializer=dum)
       return -tf.reduce_mean(w)
        

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != len(self.y_pred.shape):
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype:
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return reduce((lambda p,q: p+q),[1 for i in range(len(y)) if self.y_pred[i]!=y[i]])/len(y)
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