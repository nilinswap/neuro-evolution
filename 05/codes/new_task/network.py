import theano
import theano.tensor as T

class Network:
	
	def __init__(self,inputdim,outputdim,trainarr,testarr):
		trainx=trainarr[:inputdim,:]
		trainy=trainarr[inputdim:,:]
		testx=testarr[:inputdim,:]
		testy=testarr[inputdim:,:]

		self.inputdim=inputdim
		self.outputdim=outputdim
		self.trainx = theano.shared(
            value=trainx,
            name='trainx',
            borrow=True
        )
        self.trainy = theano.shared(
            value=trainy,
            name='trainy',
            borrow=True
        )
        self.testx = theano.shared(
            value=testx,
            name='testx',
            borrow=True
        )
        self.testy = theano.shared(
            value=testy,
            name='testy',
            borrow=True
        )

    def feedforward(hid_nodes,weight_arr):
    	fir_weight=weight_arr[:(self.inputdim+1)*hid_nodes].reshape(self.inputdim+1,hid_nodes)
    	sec_weight=weight_arr[(self.inputdim+1)*hid_nodes:].reshape((hid_nodes+1),self.outputdim)
    	

