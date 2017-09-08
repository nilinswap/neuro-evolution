import numpy as np
import network

class Population(object):
	"""Class to create population object, and handle its methods"""
	list_chromo = []
	fits_pops = []

	def __init__(self, D, size,net,limittup=(-1,1)):
		self.D = D
		self.size = size
		self.net=net
		self.list_chromo = np.random.uniform(limittup[0],limittup[1],(self.size,(self.net.inputdim+1)*self.net.hid_nodes+(self.net.hid_nodes+1)*self.net.outputdim))#a numpy array
		self.fits_pops=[]
	
	def set_list_chromo(self,newlist_chromo):
		p=self.list_chromo
		self.list_chromo=newlist_chromo# ndarray
		self.set_fitness()
		del(p)

	def set_fitness(self):
		fitness_func=self.net.feedforward
		self.fits_pops=fitness_func(self.list_chromo)#another np array

	def get_best(self):
		if not len(self.fits_pops):
			self.set_fitness()
		min_ind=np.argmin(self.fits_pops)
		return (self.list_chromo[min_ind],self.fits_pops[min_ind])

	def get_average(self):
		if not len(self.fits_pops):
			self.set_fitness()
		return np.mean(self.fits_pops)

def squa_test(x):
	return (x**2).sum(axis=1)

def main():	
	import copy
	trainarr = np.concatenate((np.arange(0,9).reshape(3,3),np.array([[1,0],[0,1],[1,0]])),axis=1)
	testarr = copy.deepcopy(trainarr)
	#print(net.trainx,net.trainy)
	hid_nodes = 4
	indim = 3
	outdim = 2
	size = 5
	net = network.Network(indim,outdim,hid_nodes,trainarr,testarr)
	arr_of_net = np.random.uniform(-1,1,(size,(indim+1)*hid_nodes+(hid_nodes+1)*outdim))
	#print(arr_of_net)
	#print(net.feedforward(hid_nodes,arr_of_net))
	pop = Population(3,5,net)
	#pop.set_list_chromo(arr_of_net)

	print(pop.list_chromo)
	pop.set_fitness()
	print(pop.fits_pops)
	print(pop.get_best())
	print(pop.get_average())

if __name__=='__main__':
	main()
