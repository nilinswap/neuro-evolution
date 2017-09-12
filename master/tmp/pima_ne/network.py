#network.py
import numpy as np
import theano
import tmlp
import theano.tensor as T
rng=np.random
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
	def __init__(self,inputdim,outputdim,arr_of_net,trainx,trainy,testx,testy):
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
		self.arr_of_net=arr_of_net

	def feedforward(self):
		#weight_arr = np.array(weight_arr)
		#arr_of_net  type: nd.array, it is a whole list of network (i.e. each vector is a new network with hid_nodes)
		lis=[]

		for i in range(self.arr_of_net.shape[0]):
			hid_nodes=int(self.arr_of_net[i][0])
			fir_weight = self.arr_of_net[i][1:((self.inputdim+1)*hid_nodes)+1].reshape(self.inputdim+1, hid_nodes)
			sec_weight = self.arr_of_net[i][((self.inputdim+1)*hid_nodes)+1:].reshape((hid_nodes+1), self.outputdim)
			trainx = np.concatenate((self.trainx,-np.ones((self.trainx.shape[0],1))),axis=1)
			midout = np.dot(trainx,fir_weight)
			midout = sigmoid(midout)
			midout = np.concatenate((midout, -np.ones((midout.shape[0],1))), axis=1)
			output = np.dot(midout, sec_weight)
			output = sigmoid(output)
			er_arr = (1/2)*np.mean((output-self.trainy)**2)
			lis.append(er_arr)
		return np.array(lis)

	def test(self,weight_arr):
		hid_nodes=int(weight_arr[0])
		fir_weight = weight_arr[1:(self.inputdim+1)*hid_nodes+1].reshape(self.inputdim+1,hid_nodes)
		sec_weight = weight_arr[(self.inputdim+1)*hid_nodes+1:].reshape((hid_nodes+1),self.outputdim)
		testx = np.concatenate((self.testx,-np.ones((self.testx.shape[0],1))),axis=1)
		midout = np.dot(testx,fir_weight)
		midout = sigmoid(midout)
		midout = np.concatenate((midout,-np.ones((midout.shape[0],1))),axis=1)
		output = np.dot(midout,sec_weight)
		output = sigmoid(output)
		er_arr = (1/2)*np.mean((output-self.testy)**2)
		return er_arr

	def modify_thru_backprop(popul):
		"""lis_of_keys=list(popul.k_dict.keys())
		for hid_nodes in lis_of_keys:
			newfullnet=Backnet(hid_nodes,popul.net_err)
			for ind in popul.k_dict[hid_nodes]:
				newfullnet.set_weights(popul.list_chromo[ind])
				newfullnet.train()
				popul.list_chromo[ind]=newfullnet.get_new_weight()"""
		lis=[]
		for i in range(popul.size):
			lis.append(Backnet(i,popul.net_err))#here backnet should return a new numpy 1d array
		
		popul.set_list_chromo(np.array(lis))
		#popul.set_fitness()
def Backnet(ind,net_err,n_par=10,n_epochs=300):
	weight_str=net_err.arr_of_net[ind]
	hid_nodes=int(weight_str[0])
	
	outputdim=2####	COMPLETE SHIT BUT CAN'T HELP
	inputdim=net_err.inputdim
	
	trainx=net_err.trainx
	trainy=net_err.trainy
	##very much 'net_err dependent the below part is , if net_err's trainy was a 1d array, below would be different
	#print("in backnet",trainy)
	if trainy.shape[1]==1:
		newtrainy=np.zeros((trainy.shape[0],2))
		for i in range(trainy.shape[0]):
			newtrainy[i][int(trainy[i][0])]=1

	else:
		newtrainy=trainy
	rest_setx,rest_sety=shared_dataset((trainx,newtrainy))#returns shared variable
	del(trainx)
	del(trainy)
	del(newtrainy)
	#here rest_sety, due to casting it is converted into tensor_variable, can only be accessed using function.
	#fun=theano.function([],rest_sety)
	#print(rest_setx.get_value(),fun())
	par_size=int(rest_setx.get_value().shape[0]//n_par)
	#print(par_size)
	prmsdind=T.lscalar()
	
	
	valid_x_to_be=rest_setx[prmsdind*par_size:(prmsdind+1)*par_size,:]
	valid_y_to_be=rest_sety[prmsdind*par_size:(prmsdind+1)*par_size]
	train_x_to_be=T.concatenate((rest_setx[:(prmsdind)*par_size,:],rest_setx[(prmsdind+1)*par_size:,:]),axis=0)
	train_y_to_be=T.concatenate((rest_sety[:(prmsdind)*par_size],rest_sety[(prmsdind+1)*par_size:]))
	fun=theano.function([prmsdind],valid_y_to_be)
	#print("here",fun(0))
	x = T.matrix('x')  # the data is presented as rasterized images
	y = T.ivector('y')

	classifier= tmlp.MLP(
		    rng=rng,
		    input=x,
		    n_in=8,
		    n_hidden=hid_nodes,
		    n_out=2,			#this was important, I tried taking 1 gave fatal results, decided to fix this later.
			weight_str=weight_str #unaltered string
		)
	cost = (
		    classifier.mean_square_error(y)
		    
		)
	validate_model = theano.function(
	    inputs=[prmsdind],
	    outputs=classifier.mean_square_error(y),
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

	for i in range(n_epochs):
		avrg=0
		for ind in range(n_par):
			trainerr=train_model(ind)
		if i%10==0:
			validerr=validate_model(ind)
			avrg=(avrg*(ind)+validerr)/(ind+1)
			print("		in training",i,avrg)
	fir_weight=classifier.fir_weight.get_value()
	fir_b=classifier.fir_b.get_value()
	sec_weight=classifier.sec_weight.get_value()
	sec_b=classifier.sec_b.get_value()
	fullfirwei=np.concatenate((fir_weight,fir_b),axis=0).reshape(((inputdim+1)*hid_nodes,))
	fullsecwei=np.concatenate((sec_weight,sec_b),axis=0).reshape(((hid_nodes+1)*ouptutdim,))

	w_str=[float(hid_nodes)]+list(fullfirwei)+list(fullsecwei)
	return np.array(w_str)


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
