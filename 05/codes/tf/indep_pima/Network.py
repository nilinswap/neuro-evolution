import numpy as np
#import theano
import tf_mlp
import tensorflow as tf
import time

def sigmoid(arr):
	return 1/(1+np.exp(-arr))
class Neterr:	
	def __init__(self,inputdim,outputdim,arr_of_net,rest_setx,rest_sety,test_setx,test_sety,rng,n_par=10):
		"""trainx=trainarr[:,:inputdim]
		trainy=trainarr[:,inputdim:]
		testx=testarr[:,:inputdim]
		testy=testarr[:,inputdim:]
		"""
		self.inputdim=inputdim
		self.outputdim=outputdim
		self.srest_setx=rest_setx
		self.srest_sety=rest_sety
		self.stest_setx=test_setx
		self.stest_sety=test_sety
		self.rng=rng
		self.x=tf.placeholder(name='x',dtype=tf.float64,shape=[None,self.inputdim])
		self.y=tf.placeholder(name='y',dtype=tf.int32,shape=[None,])
		savo1=tf.train.Saver(var_list=[self.srest_setx,self.srest_sety,self.stest_setx,self.stest_sety])
		with tf.Session() as sess:
			savo1.restore(sess, "/home/placements2018/forgit/neuro-evolution/05/state/tf/indep_pima/input/model.ckpt")		#only restored for this session
			self.trainx=self.srest_setx.eval()
			self.trainy=self.srest_sety.eval()
			self.testx=self.stest_setx.eval()
			self.testy=self.stest_sety.eval()
		self.n_par=n_par
		par_size=int(self.trainx.shape[0]/n_par)
		self.prmsdind=tf.placeholder(name='prmsdind',dtype=tf.int32)
		self.valid_x_to_be=self.srest_setx[self.prmsdind*par_size:(self.prmsdind+1)*par_size,:]
		self.valid_y_to_be=self.srest_sety[self.prmsdind*par_size:(self.prmsdind+1)*par_size]
		self.train_x_to_be=tf.concat((self.srest_setx[:(self.prmsdind)*par_size,:],self.srest_setx[(self.prmsdind+1)*par_size:,:]),axis=0)
		self.train_y_to_be=tf.concat((self.srest_sety[:(self.prmsdind)*par_size],self.srest_sety[(self.prmsdind+1)*par_size:]),axis=0)
		
		
		self.arr_of_net=arr_of_net
		#

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
			midout = np.tanh(midout)
			midout = np.concatenate((midout, -np.ones((midout.shape[0],1))), axis=1)
			output = np.dot(midout, sec_weight)
			output = sigmoid(output)
			er_arr = (1/2)*np.mean((output-self.trainy)**2)
			"""import copy
			p=copy.deepcopy(output)

			q=copy.deepcopy(self.trainy)
			p=p.ravel()
			q=q.ravel()
			er_arr=np.mean(-q*np.log(p))"""
			lis.append(er_arr)
		return np.array(lis)

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

	def modify_thru_backprop(self,popul,epochs=10,learning_rate=0.01,L1_reg=0.00001,L2_reg=0.0001):
		
		
		lis=[]
		lis_of_keys=list(popul.k_dict.keys())

		for hid_nodes in lis_of_keys:
			if hid_nodes not in popul.net_dict:
				newmlp=tf_mlp.MLP(self.x,self.inputdim,self.outputdim,hid_nodes,self.rng)
				popul.net_dict[hid_nodes]=newmlp

			fullnet=popul.net_dict[hid_nodes]
			
			cost = tf.add(tf.add(
			    fullnet.negative_log_likelihood(self.y)
			    ,L1_reg * fullnet.L1
			    ),L2_reg * fullnet.L2_sqr
			)
			
			optmzr = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
			savo1=tf.train.Saver(var_list=[self.srest_setx,self.srest_sety,self.stest_setx,self.stest_sety])
			with tf.Session() as sess:
		
				savo1.restore(sess, "/home/placements2018/forgit/neuro-evolution/05/state/tf/indep_pima/input/model.ckpt")
				sess.run([fullnet.logRegressionLayer.W.initializer,fullnet.logRegressionLayer.b.initializer,fullnet.hiddenLayer.W.initializer,fullnet.hiddenLayer.b.initializer])
				#print("------------------------------------------------------------------")
				#print(sess.run([valid_x_to_be,valid_y_to_be,train_x_to_be,train_y_to_be],feed_dict={self.prmsdind:0}))
				
				#print(sess.run([cost],feed_dict={x:train_x_to_be.eval(feed_dict={self.prmsdind:zhero}),y:train_y_to_be.eval(feed_dict={self.prmsdind:zhero})}))
				
				#cool thing starts from here ->
			    ######################
			    # BUILD ACTUAL MODEL #
			    ######################

				print('...building the model')
				print("nhid", hid_nodes)
				for pind in popul.k_dict[hid_nodes]:
					print("switch")
					fullnet.set_weights_from_chromosome(popul.list_chromo[pind])
					prevtoprev=10#just any no. which does not satisfy below condition
					prev=7
					current=5
					for epoch in range(epochs):
						listisi=[]
						for ind in range(self.n_par):
							_,bost=sess.run([optmzr,cost],feed_dict={self.x:self.train_x_to_be.eval(feed_dict={self.prmsdind:ind}),self.y:self.train_y_to_be.eval(feed_dict={self.prmsdind:ind})})
				
							if epoch%(epochs//4)==0:

								q=fullnet.errors(self.y).eval(feed_dict={self.x:self.valid_x_to_be.eval(feed_dict={self.prmsdind:ind}),self.y:self.valid_y_to_be.eval(feed_dict={self.prmsdind:ind})})
								listisi.append(q)
						if epoch%(epochs//4)==0:
							
							prevtoprev=prev
							prev=current
							current=np.mean(listisi)
							print('validation',current,prevtoprev,prev)
						
						if prev-current <0.002 and prevtoprev-prev<0.002:
							break;
								

					lis.append(fullnet.turn_weights_into_chromosome())	
				

			
			

		
		popul.set_list_chromo(np.array(lis))
		
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
