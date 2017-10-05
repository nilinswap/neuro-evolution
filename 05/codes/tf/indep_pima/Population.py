import numpy as np
import Network
import pimadataf
import tensorflow as tf
import os
def givesumar(size):
		ar=[0]
		for i in range(1,size+1):
			ar+=[ar[i-1]+i]
		return ar

class Population(object):
	"""Class to create population object, and handle its methods"""

	def __init__(self,rng, max_hidden_units, size=5, limittup=(-1,1)):
		self.dimtup = pimadataf.get_dimension()
		rest_set, test_set = pimadataf.give_data()
		restx=rest_set[0]
		resty=rest_set[1]
		testx=test_set[0]
		testy=test_set[1]
		resty=np.ravel(resty)
		testy=np.ravel(testy)
		self.rng=rng
		self.size = size
		self.max_hidden_units = max_hidden_units
		self.list_chromo = self.aux_pop(size, limittup) #a numpy array
		self.fits_pops = []
		restn=538										#a flaw here ,one has to know no. of datapoints in both set before opening it(inside program)
		testn=230		
		print("here you",rest_set[1].shape)
		self.rest_setx=tf.Variable(initial_value=np.zeros((restn,self.dimtup[0])),name='rest_setx',dtype=tf.float64)
		self.rest_sety=tf.Variable(initial_value=np.zeros((restn,)),name='rest_sety',dtype=tf.int32)
		self.test_setx=tf.Variable(initial_value=np.zeros((testn,self.dimtup[0])),name='rest_sety',dtype=tf.float64)
		self.test_sety=tf.Variable(initial_value=np.zeros((testn,)),name='test_sety',dtype=tf.int32)
		if not os.path.isfile('/home/robita/forgit/neuro-evolution/05/state/tf/indep_pima/input/model.ckpt.meta'):
			

			

			rxn=self.rest_setx.assign(restx)
			ryn=self.rest_sety.assign(resty)
			txn=self.test_setx.assign(testx)
			tyn=self.test_sety.assign(testy)
			var_lis=[self.rest_setx,self.rest_sety,self.test_setx,self.test_sety]
			nodelis=[rxn,ryn,txn,tyn]
			savo=tf.train.Saver(var_list=var_lis)
			with tf.Session() as sess:
				sess.run([i for i in nodelis])
				print("saving checkpoint")
				save_path = savo.save(sess, "/home/robita/forgit/neuro-evolution/05/state/tf/indep_pima/input/model.ckpt")

		
		
		
		self.net_err = Network.Neterr(inputdim=self.dimtup[0], outputdim=self.dimtup[1], arr_of_net=self.list_chromo,rest_setx=self.rest_setx,rest_sety=self.rest_sety,test_setx=self.test_setx,test_sety=self.test_sety,rng=self.rng)
		self.net_dict={} #dictionary of networks for back-propagation, one for each n_hid
	
	def create_dict(self):
		k_dict = {}
		par = list(-self.fits_pops)
		ar = np.arange(0,self.size)
		sumar = [0]
		
		for i in range(1, self.size+1):
			sumar.append(sumar[i-1]+i)
		self.sumar = sumar
		listup = list(zip(list(ar),par))
		listup.sort(key=lambda x: x[1])
		self.sortedlistup = listup
		sum_dict = {}

		for i in range(len(self.list_chromo)):
			if int(self.list_chromo[i][0]) in k_dict:
				k_dict[int(self.list_chromo[i][0])].append(i)
			else:
				k_dict[int(self.list_chromo[i][0])]=[i]
		
		for lis in k_dict.values():
			lis.sort(key=lambda x: -self.fits_pops[x])
		
		for k in k_dict.keys():
			sum_dict[k]=givesumar(len(k_dict[k]))
		
		self.sum_dict=sum_dict
		self.k_dict=k_dict

	def aux_pop(self, size,limittup):	
		population = []
		inputdim = self.dimtup[0]
		outputdim = self.dimtup[1]
		for i in range(1, self.max_hidden_units+1):
			for j in range(size//self.max_hidden_units):
				population.append(np.concatenate([[i], self.rng.uniform(limittup[0],limittup[1], ((inputdim+1)*i + (i+1)*outputdim))]))

		for i in range(1,size%self.max_hidden_units+1):
			population.append(np.concatenate([[i], self.rng.uniform(limittup[0], limittup[1], ((inputdim+1)*i + (i+1)*outputdim))]))
		return np.array(population)

	def set_list_chromo(self,newlist_chromo):
		p = self.list_chromo
		self.list_chromo = newlist_chromo# ndarray
		
		self.set_fitness()
		del(p)

	def set_fitness(self):
		#del(self.net_err)
		#self.net_err = Network.Neterr(inputdim=self.dimtup[0], outputdim=self.dimtup[1], arr_of_net=self.list_chromo,rest_setx=self.rest_setx,rest_sety=self.rest_sety,test_setx=self.test_setx,test_sety=self.test_sety,rng=self.rng)
		self.net_err.set_arr_of_net(self.list_chromo)
		fitness_func = self.net_err.feedforward
		self.fits_pops = fitness_func()#another np array
		self.create_dict()

	def get_best(self):
		if not len(self.fits_pops):
			self.set_fitness()
		min_ind = np.argmin(self.fits_pops)
		return (self.list_chromo[min_ind], self.fits_pops[min_ind],min_ind)#error alert!!

	def get_average(self):
		if not len(self.fits_pops):
			self.set_fitness()
		return np.mean(self.fits_pops)

def squa_test(x):
	return (x**2).sum(axis=1)

def main():	
	import copy
	dimtup=(8,1)
	pop=Population(4,dimtup,size=9)

	print(pop.list_chromo)
	pop.set_fitness()
	print(pop.fits_pops)
	print(pop.k_dict)
	print(pop.sortedlistup)
	print(pop.sumar)
	print(pop.sum_dict)
	neter = Network.Neterr(dimtup[0],dimtup[1],pop.list_chromo,pop.trainx,pop.trainy,pop.testx,pop.testy)
	Network.Backnet(4,neter)

if __name__=='__main__':
	main()
