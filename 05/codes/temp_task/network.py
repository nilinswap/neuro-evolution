#network.py
def sigmoid(arr):
	return 1/(1+np.exp(-arr))
class Network:
	
	def __init__(self,inputdim,outputdim,trainarr,testarr):
		trainx=trainarr[:,:inputdim]
		trainy=trainarr[:,inputdim:]
		testx=testarr[:,:inputdim]
		testy=testarr[:,inputdim:]

		self.inputdim=inputdim
		self.outputdim=outputdim
		self.trainx = trainx
		self.trainy = trainy
		self.testx = testx
		self.testy = testy

	def feedforward(self,hid_nodes,weight_arr):
		weight_arr=np.array(weight_arr)
		fir_weight=weight_arr[:(self.inputdim+1)*hid_nodes].reshape(self.inputdim+1,hid_nodes)
		sec_weight=weight_arr[(self.inputdim+1)*hid_nodes:].reshape((hid_nodes+1),self.outputdim)
		trainx=np.concatenate((self.trainx,-np.ones((self.trainx.shape[0],1))),axis=1)
		midout=np.dot(trainx,fir_weight)
		midout=sigmoid(midout)
		midout=np.concatenate((midout,-np.ones((midout.shape[0],1))),axis=1)
		output=np.dot(midout,sec_weight)
		output=sigmoid(output)
		er_arr=(1/2)*np.mean((output-self.trainy)**2)
		return er_arr
	def test(self,hid_nodes,weight_arr):
		fir_weight=weight_arr[:(self.inputdim+1)*hid_nodes].reshape(self.inputdim+1,hid_nodes)
		sec_weight=weight_arr[(self.inputdim+1)*hid_nodes:].reshape((hid_nodes+1),self.outputdim)
		testx=np.concatenate((self.testx,-np.ones((self.testx.shape[0],1))),axis=1)
		midout=np.dot(testx,fir_weight)
		midout=sigmoid(midout)
		midout=np.concatenate((midout,-np.ones((midout.shape[0],1))),axis=1)
		output=np.dot(midout,sec_weight)
		output=sigmoid(output)
		er_arr=(1/2)*np.mean((output-self.testy)**2)
		return er_arr