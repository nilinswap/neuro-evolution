import theano
import theano.tensor as T
from theano.tensor.nnet import nnet
import numpy as np
rng=np.random

class tpcn:
	def __init__(self,inputh,inpdim,outdim):
		
		self.inpdim=inpdim
		self.outdim=outdim
		self.input=inputh# this is T.tensor type
		self.w=theano.shared(value=np.zeros((inpdim,outdim),dtype=theano.config.floatX),
			name="w",borrow=True)
		self.b=theano.shared(value=np.zeros((outdim,),dtype=theano.config.floatX),name="b",borrow=True)

		self.y = T.nnet.softmax(T.dot(self.input,self.w) + self.b)
		self.y=self.y.argmax(axis=1)	
		#self.cost = T.nnet.categorical_crossentropy(self.y, tar).mean()#check here
		#gw, gb = T.grad(cost, [w, b])
		#eta=0.2

	def cost(self,tar)
		return T.nnet.categorical_crossentropy(self.y, tar).mean()
    def errors(self, tar):
       

        # check if y has same dimension of y_pred
        if tar.ndim != self.y.ndim:
            raise TypeError(
                'tar should have the same shape as self.y',
                ('y', tar.type, 'y_pred', self.y.type)
            )
        # check if y is of the correct datatype
        if tar.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y,tar ))
        else:
            raise NotImplementedError()	

	def tconfmat(y,predictions):# both y and predictions are vector of P
		
		# Add the inputs that match the bias node
		
	
		nClasses = self.outdim

		if nClasses==1:
			nClasses = 2
			outputs = np.where(predictions>0,1,0)
			targets=y
		else:
			# 1-of-N encoding
			outputs = np.argmax(predictions,1)
			targets = np.argmax(y,1)

		cm = np.zeros((nClasses,nClasses))
		for i in range(nClasses):
			for j in range(nClasses):
				cm[i,j] = np.sum(np.where(outputs==i,1,0)*np.where(targets==j,1,0))

		print (cm)
		err= 1-(np.trace(cm)/np.sum(cm)) 
		print(err)
		return err
