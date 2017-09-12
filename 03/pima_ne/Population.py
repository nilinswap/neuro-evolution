import numpy as np
import network

class Population(object):
	"""Class to create population object, and handle its methods"""
	
	def __init__(self, D, size,limittup=(-1,1)):
		self.D = D
		self.size = size
		self.list_chromo = self.aux_pop(size,limittup, 16) 
		print(self.list_chromo)
		self.fits_pops=[]

	def create_dict(self):
		k_dict = {}
		sum_dict={}
		sum_fit=[0]
		for i in range(len(self.list_chromo)):
			sum_fit.append(sum(sum_fit)+self.fits_pops[i])
			if self.list_chromo[i][0] in k_dict:
				k_dict[self.list_chromo[i][0]].append(i)
				sum_dict[self.list_chromo[i][0]].append(sum(sum_dict[self.list_chromo[i][0]])+self.fits_pops[i])	
			else:
				k_dict[self.list_chromo[i][0]]=[i]
				sum_dict[self.list_chromo[i][0]]=[0,self.fits_pops[i]]
	
	def set_list_chromo(self,newlist_chromo):
		p = self.list_chromo
		self.list_chromo=newlist_chromo# ndarray
		self.set_fitness()
		del(p)

	def set_fitness(self):
		fitness_func = self.net.feedforward
		self.fits_pops = fitness_func() #another np array
		self.create_dict()

	def get_best(self):
		if not len(self.fits_pops):
			self.set_fitness()
		min_ind=np.argmin(self.fits_pops)
		return (self.list_chromo[min_ind],self.fits_pops[min_ind])

	def get_average(self):
		if not len(self.fits_pops):
			self.set_fitness()
		return np.mean(self.fits_pops)

	def aux_pop(self, size,limittup, no_of_hidden_units):	
		population = []
		inputdim = 3
		outputdim =2
		for i in range(1,no_of_hidden_units+1):
			for j in range(size//no_of_hidden_units):
				population.append(np.concatenate([[i],np.random.uniform(limittup[0],limittup[1],((inputdim+1)*i + (i+1)*outputdim))]))

		for i in range(1,size%no_of_hidden_units+1):
			population.append(np.concatenate([[i],np.random.uniform(limittup[0],limittup[1],((inputdim+1)*i + (i+1)*outputdim))]))
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
	
	#print(arr_of_net)
	#print(net.feedforward(hid_nodes,arr_of_net))
	pop = Population(3,5)
	#pop.set_list_chromo(arr_of_net)

	print(pop.list_chromo)
	pop.set_fitness()
	print(pop.fits_pops)
	print(pop.k_dict)
	print(pop.get_best())
	print(pop.get_average())

if __name__=='__main__':
	main()
