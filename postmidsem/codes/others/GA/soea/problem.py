#problem class
import numpy as np
def testfun(arr):	#np.array of 1 dim for one vector. 
	
	return arr[0]

def  hcef(arr):
	n=len(arr)
	lis=[((i-1)/(n-1))*arr[i-1]**2 for i in range(1,n+1)] 
	return sum(lis)

def rastrigin(arr):
	n=len(arr)
	lis=[arr[i-1]**2-10*np.cos(2*np.pi*arr[i-1])+10 for i in range(1,n+1)] 
	return sum(lis)

def rastrigint(arr):
	n=len(arr)
	lis=[(arr[i-1]-4)**2-10*np.cos(2*np.pi*(arr[i-1]-4))+10 for i in range(1,n+1)] 
	return sum(lis)

def katsuura(arr):
	n=len(arr)
	po=10/n**1.2
	bigsumi=1
	for i in range(1,n+1):
		sumi=0
		for j in range(1,33):
			sumi+=(abs(2**j*arr[i-1]-round(2**j*arr[i-1])))/2**j
		bigsumi*=(1+i*sumi)
	bigsumi*=pow(bigsumi,po)
	return (10/n**2)*(bigsumi-1)

class Problem:
	def __init__(self,fitness_func=testfun,dim=1,prangetup=None,constraints=None,opttype=0,expmax=10000):
		self.fitness=fitness_func	#function variable i.e. variable refrencing to a function
		self.inputdim=dim	
		self.rangetup=prangetup			#a tuple of form ( start, end)
		self.constraints=constraints #type yet to be decided
		self.opttype=opttype	#i.e. minimize or maximize 
		self.expmax=expmax 		#i.e. expected maximum to be used in find expectation


	def find_fitness(self,xarr):#xarr  is  vector   type
		return self.fitness(xarr)







	
