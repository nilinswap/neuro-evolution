import numpy as np
import network
import pimadataf

def givesumar(size):
		ar=[0]
		for i in range(1,size+1):
			ar+=[ar[i-1]+i]
		return ar

class Population(object):
	"""Class to create population object, and handle its methods"""

	def __init__(self, max_hidden_units, size=5, limittup=(-1,1)):
		self.dimtup = pimadataf.get_dimension()
		rest_set, test_set = pimadataf.give_data()
		tup = pimadataf.give_datainshared()

		self.size = size
		self.max_hidden_units = max_hidden_units
		self.list_chromo = self.aux_pop(size, limittup) #a numpy array
		self.fits_pops = []
				
		self.trainx = rest_set[0]
		self.trainy = rest_set[1]
		self.testx = test_set[0]
		self.testy = test_set[1]
		
		self.strainx, self.strainy = tup[0]
		self.stestx, self.stesty = tup[1]
		self.net_err = network.Neterr(inputdim=self.dimtup[0], outputdim=self.dimtup[1], arr_of_net=self.list_chromo, trainx=self.trainx, trainy=self.trainy, testx=self.testx, testy=self.testy,strainx=self.strainx, strainy=self.strainy, stestx=self.stestx, stesty=self.stesty)
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
			#sum_fit.append(sum(sum_fit)+self.fits_pops[i])
			if int(self.list_chromo[i][0]) in k_dict:
				k_dict[int(self.list_chromo[i][0])].append(i)
				#sum_dict[int(self.list_chromo[i][0])].append(sum(sum_dict[int(self.list_chromo[i][0])])+self.fits_pops[i])	
			else:
				k_dict[int(self.list_chromo[i][0])]=[i]
				#sum_dict[int(self.list_chromo[i][0])]=[0,self.fits_pops[i]]
		
		for lis in k_dict.values():
			lis.sort(key=lambda x: -self.fits_pops[x])
		for k in k_dict.keys():
			sum_dict[k]=givesumar(len(k_dict[k]))
		self.sum_dict=sum_dict
		self.k_dict=k_dict

		#self.sum_dict=sum_dict
		#self.sum_fit=sum_fit

	def aux_pop(self, size,limittup):	
		population = []
		inputdim=self.dimtup[0]
		outputdim=self.dimtup[1]
		for i in range(1,self.max_hidden_units+1):
			for j in range(size//self.max_hidden_units):
				population.append(np.concatenate([[i],np.random.uniform(limittup[0],limittup[1],((inputdim+1)*i + (i+1)*outputdim))]))

		for i in range(1,size%self.max_hidden_units+1):
			population.append(np.concatenate([[i],np.random.uniform(limittup[0],limittup[1],((inputdim+1)*i + (i+1)*outputdim))]))
		return np.array(population)
	def set_list_chromo(self,newlist_chromo):
		p=self.list_chromo
		self.list_chromo=newlist_chromo# ndarray
		self.set_fitness()
		del(p)

	def set_fitness(self):
		self.net_err=network.Neterr(inputdim=self.dimtup[0],outputdim=self.dimtup[1],arr_of_net=self.list_chromo,trainx=self.trainx,trainy=self.trainy,testx=self.testx,testy=self.testy,strainx=self.strainx,strainy=self.strainy,stestx=self.stestx,stesty=self.stesty)
		fitness_func=self.net_err.feedforward
		self.fits_pops=fitness_func()#another np array
		#print(self.fits_pops)
		
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
	neter = network.Neterr(dimtup[0],dimtup[1],pop.list_chromo,pop.trainx,pop.trainy,pop.testx,pop.testy)
	network.Backnet(4,neter)
	#print(pop.get_best())
	#print(pop.get_average())

if __name__=='__main__':
	main()
