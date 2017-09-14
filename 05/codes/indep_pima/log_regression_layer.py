import numpy as np 
import theano
import theano.tensor as T 

class LogRegressionLayer:
    def __init__(self,input,n_in,n_out,W=None,b=None):

       
        if not W:
            self.W = theano.shared(
                value=np.zeros(
                    (n_in, n_out),
                    dtype=theano.config.floatX
                ),
                name='W',
                borrow=True
            )
        else:
            self.W=W
        if not b:
        # initialize the biases b as a vector of n_out 0s
            self.b = theano.shared(
                value=np.zeros(
                    (n_out,),
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )
        else:
            self.b=b

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        if n_out!=1:
        	self.y_pred=T.argmax(self.p_y_given_x, axis=1)
        else:
        	self.y_pred=self.p_y_given_x
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input