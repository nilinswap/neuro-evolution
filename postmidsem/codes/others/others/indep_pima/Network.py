import numpy as np
import theano
import tmlp
import theano.tensor as T
import tmlp
import time

def sigmoid(arr):
	return 1/(1+np.exp(-arr))
def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
       
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')
class Neterr:	
	def __init__(self,inputdim,outputdim,trainx,trainy,testx,testy,strainx,strainy,stestx,stesty):
		"""trainx=trainarr[:,:inputdim]
		trainy=trainarr[:,inputdim:]
		testx=testarr[:,:inputdim]
		testy=testarr[:,inputdim:]
		"""
		self.inputdim=inputdim
		self.outputdim=outputdim
		self.trainx = trainx
		self.trainy = trainy
		self.testx = testx
		self.testy = testy
		
		self.strainx = strainx
		self.strainy = strainy
		self.stestx = stestx
		self.stesty = stesty
		#self.arr_of_net=arr_of_net
		self.fundam=theano.function([],self.strainy)


	def test(self,weight_arr):
		hid_nodes=int(weight_arr[0])
		fir_weight = weight_arr[1:(self.inputdim+1)*hid_nodes+1].reshape(self.inputdim+1,hid_nodes)
		sec_weight = weight_arr[(self.inputdim+1)*hid_nodes+1:].reshape((hid_nodes+1),self.outputdim)
		testx = np.concatenate((self.testx,-np.ones((self.testx.shape[0],1))),axis=1)
		midout = np.dot(testx,fir_weight)
		midout = np.tanh(midout)
		midout = np.concatenate((midout,-np.ones((midout.shape[0],1))),axis=1)
		output = np.dot(midout,sec_weight)
		output = sigmoid(output)
		for i in range(len(output)):
			if output[i]>0.5:
				output[i]=1
			else:
				output[i]=0
		er_arr=np.mean(abs(output-self.testy))
		#er_arr = (1/2)*np.mean((output-self.testy)**2)
		
		return er_arr

	def modify_thru_backprop(self,epochs=100,learning_rate=0.01):
		
		y=T.ivector('y')
		lis=[]
		#lis_of_keys=list(popul.k_dict.keys())
		hid_nodes = 100
		newmlp=tmlp.MLP(self.inputdim,self.outputdim,hid_nodes,self.strainx,self.strainy,self.stestx,self.stesty)


		fullnet=newmlp
		params=fullnet.params
		cost=fullnet.cost_func(y)

		gparams=[T.grad(cost, j) for j in params]
		updates = [
					(param, param - learning_rate * gparam)
					for param, gparam in zip(params, gparams)
				]

		train_model=theano.function([],cost,updates=updates,givens={y:self.strainy},on_unused_input='ignore')
		#fun1=theano.function([x],output.reshape((x.shape[0],)))
		#
		test_model=theano.function([],fullnet.find_error(y),givens={y:self.stesty},on_unused_input='ignore')




		for i in range(1,epochs):
			#p=train_model(rest_setx.get_value(),fun())

			p=train_model()

			print("in back  training",i,hid_nodes,p)

		print("here sub testing",test_model())
		print(self.test(fullnet.turn_weights_into_chromosome()))

		"""lis=[]
		for i in range(popul.size):
			lis.append(Backnet(i,popul.net_err))#here backnet should return a new numpy 1d array
		"""

		
		#popul.set_fitness()



def squa_test(x):
	return (x**2).sum(axis=1)

def main():
	#print("hi")
	import copy
	"""trainarr = np.concatenate((np.arange(0,9).reshape(3,3),np.array([[1,0],[0,1],[1,0]])),axis=1)
	testarr = copy.deepcopy(trainarr)
	trainx=trainarr[:,:3]
	trainy=trainarr[:,3:]
	testx=testarr[:,:3]
	testy=testarr[:,3:]
	hid_nodes = 4
	indim = 3
	outdim = 2
	size = 5
	"""
	hid_nodes = 4
	indim = 10
	outdim = 1
	size = 100
	resularr=np.zeros((size,outdim))
	for i in range(size):
		#resularr[i][np.random.randint(0,outdim)]=1
		if np.random.randint(0,2)==1:
			resularr[i][0]=1
	#resularr
	trainarr = np.concatenate((np.arange(0,1000).reshape(100,10),resularr),axis=1)
	testarr = copy.deepcopy(trainarr)
	trainx=trainarr[:,:indim]
	trainy=trainarr[:,indim:]
	testx=testarr[:,:indim]
	testy=testarr[:,indim:]
	#arr_of_net = np.random.uniform(-1,1,(size,(indim+1)*hid_nodes+(hid_nodes+1)*outdim))
	hid_nodesarr=np.random.randint(1,hid_nodes+1,size)
	lis=[]
	for i in hid_nodesarr:
		lis.append(np.concatenate((np.array([i]),np.random.uniform(-1,1,(indim+1)*i+(i+1)*outdim))))
	arr_of_net=np.array(lis)
	neter = Neterr(indim,outdim,arr_of_net,trainx,trainy,testx,testy)
	Backnet(3,neter)
	print(neter.trainx,neter.trainy)
	print(arr_of_net)
	print(neter.feedforward())
	for i in range(size):
		print(neter.test(arr_of_net[i]))
if __name__ == '__main__':
	main()
