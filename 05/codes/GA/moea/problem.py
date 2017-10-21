#problem class
import numpy as np
def testfun(arr):	#np.array of 1 dim for one vector. 
	return arr[0],arr[1]



class Problem:
	def __init__(self,prob_func,randomvec_func,constraints=None,dim=1,prangetup=None,opttype=0,expmax=10000):
		self.objectives=prob_func	#function variable i.e. variable refrencing to a function
		self.inputdim=dim	
		self.rangetup=prangetup			#a tuple of form ( start, end)
		self.constraints=constraints #type yet to be decided.........decided- a dictionary with variables and one 'rest' as key and condtion as value.
		self.opttype=opttype	#i.e. minimize or maximize 
		self.expmax=expmax 		#i.e. expected maximum to be used in find expectation
		self.randomvec_func=randomvec_func

	def find_obj(self,xarr):#xarr  is  vector   type
		return self.objectives(xarr)
	def return_random(self):
		return self.randomvec_func(self.inputdim)





	
