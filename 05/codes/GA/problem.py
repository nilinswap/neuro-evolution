#problem class
import numpy as np
def testfun(arr):	#np.array of 1 dim for one vector. 
	
	return arr[0]

def  hcef(arr):
	n=len(arr)
	lis=[((i-1)/(n-1))*10**6*arr[i-1]**2 for i in range(1,n+1)] 
	return sum(lis)

def rastrigin(arr):
	n=len(arr)
	lis=[arr[i-1]**2-10*np.cos(2*np.pi*arr[i-1])+10 for i in range(1,n+1)] 
	return sum(lis)

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







	