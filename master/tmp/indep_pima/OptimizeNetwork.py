import GeneticFunctions
import numpy as np
import random
import Population

def binsear(p,arr):
	
	low=0
	high=len(arr)-1
	while (high-low)>1:
		mid=arr[(low+high)//2]
		if mid>p:
			high=(low+high)//2
		else:
			low=(low+high)//2
	return low#this is the index of our chosen in poparr

state=0

def RoulWheel(arr):
	
	sumarr=[0]

	for i in range(len(arr)):
		sumarr.append(sumarr[i]+arr[i])
	n=len(arr)//2
	global state
	np.random.seed(state)#this was important so that the random stream does not run out .... may be 
	#RANDOM COULD BE A PROBLEM, AND ITS SEEDING. 
	state+=1
	
	for j in range(n):
		r = np.random.uniform(0,sumarr[-1],2)#gen_randuniform(0,sumarr[-1],2)
		chosenind1=binsear(r[0],sumarr)
		chosenind2=binsear(r[1],sumarr)
		yield (chosenind1,chosenind2)

def roul_wheel(sumarr):
		#r = np.random.uniform(0,sumarr[-1],1)
		#return binsear(r,fit)
		#ar=np.arange(0,size)
		#sumarr=[0]

		#for i in range(len(arr)):
		#	sumarr.append(sumarr[i]+arr[i])
		#n=len(arr)//2
		global state
		np.random.seed(state)#this was important so that the random stream does not run out .... may be 
		#RANDOM COULD BE A PROBLEM, AND ITS SEEDING. 
		state+=1
		
		
		r = np.random.uniform(0,sumarr[-1],1)#gen_randuniform(0,sumarr[-1],2)
		chosenind1=binsear(r,sumarr)
		
		return chosenind1

		
class OptimizeNetwork (GeneticFunctions.GeneticFunctions):
	def __init__(self, limit=500,switch_iter=200 , prob_crossover=0.9, prob_mutation=0.2,scale_mutation=0.33333):
		self.counter = 0
		self.limit = limit
		self.prob_crossover = prob_crossover
		self.prob_mutation = prob_mutation
		self.scale_mutation = scale_mutation
		self.switch_iter = switch_iter
		self.best = ([],np.inf, 0) #Add in class diagrams
		self.fits_pops = None

	def probability_crossover(self):
		return self.prob_crossover

	def probability_mutation(self):
		return self.prob_mutation

	def crossover(self, parents):
		father, mother = parents
		hid_nodes=father[0]
		alpha = random.uniform(0,1)
		child1 = alpha*father[1:]+(1-alpha)*mother[1:]
		child2 = alpha*mother[1:]+(1-alpha)*father[1:]
		child1=np.concatenate((np.array([hid_nodes]),child1))		
		child2=np.concatenate((np.array([hid_nodes]),child2))
		return (child1, child2)
	def RankRoulWheel(self,popul):
		ar=np.arange(0,popul.size)
		if not len(popul.fits_pops):
				popul.set_fitness()
		
		par=list(-popul.fits_pops)
		listup=list(zip(list(ar),par))
		listup.sort(key=lambda x: x[1])
		for  tup in RoulWheel(ar):
			yield popul.list_chromo[listup[tup[0]][0]],popul.list_chromo[listup[tup[1]][0]]

	
		
		
		
	def selection(self,popul):
		father=popul.list_chromo[popul.sortedlistup[roul_wheel(popul.sumar)][0]]
		#mother_i=popul.k_dict[father[0]][popul.k_dict[roul_wheel(popul.sum_dict[father[0]])]]
		mother_i=popul.k_dict[father[0]][roul_wheel(popul.sum_dict[father[0]])]
		mother = popul.list_chromo[mother_i]
		return (father, mother)

	def mutation(self, chromosome):
		mutated = chromosome
		for x in range(1,len(chromosome)):
			if np.random.random() < self.prob_mutation:
				vary = np.random.normal()*self.scale_mutation
				mutated[x] += vary
		return mutated
	
	def terminate(self,popul,nowgoback=4):
		self.counter += 1
		if self.counter%nowgoback==0:
			popul.net_err.modify_thru_backprop(popul ,epochs=100,learning_rate=0.1)#this modifies almost all the string #RTC required here
			pass
		#print("here in term",popul.list_chromo[:2])
		#f = sorted(fits_populations, reverse = True)
		f = popul.get_best()# a tuple with first being x and second being fitness
		print(f[0][0])
		if f[1] < self.best[1]:
			self.best = (f[0],f[1])

		if self.counter < self.switch_iter:
			self.prob_mutation = 0.2
		else:
			self.prob_mutation = 0.02

		if self.counter % 10 == 0:
			#fits = [f for f, ch in fits_populations]
			best = f[1]
			ave = popul.get_average()
			print(
				"[G %3d] score=(%.4f, %.4f) for %d hidden nodes" %
				(self.counter, best, ave, f[0][0]))
			#print(popul.k_dict.keys())

		if self.counter >= self.limit:
			#print("Best fitness achieved: " + str(self.best))
			#print(type(self.best[1]))
			print("sub-finally ", popul.net_err.test(f[0]))
			self.counter=0
			return True
		return False

	def run(self,popul):
		while not self.terminate(popul):
			lis=[]
			for i in range(popul.size//2):
				parent_tup=self.selection(popul)
				newborn_tup=self.crossover(parent_tup)
				child1=self.mutation(newborn_tup[0])
				child2=self.mutation(newborn_tup[1])
				lis.append(child1)
				lis.append(child2)
			#print(lis[:2])
			popul.set_list_chromo(np.array(lis))
			#print("here in func",popul.list_chromo[:2])
			#print(popul.get_average())
			#print(popul.get_best())

def main():
	import copy
	dimtup=(8,1)
	pop=Population.Population(4,dimtup,size=9)

	#print(pop.list_chromo)
	#print()
	pop.set_fitness()
	"""print(pop.fits_pops)
	print(pop.k_dict)
	print(pop.sortedlistup)
	print(pop.sumar)
	print(pop.sum_dict)"""
	on=OptimizeNetwork()
	parents=(np.array([1,2,3,4]),np.array([5,6,7,8]))
	print(on.crossover(parents))
	print(on.mutation(parents[0].astype(float)))
if __name__=="__main__":
	main()
