#problem class
def testfun(arr):	#np.array of 1 dim for one vector. 
	print("here is ",arr)
	print("with first as ",arr[0])
	return arr[0]
class Problem:
	def __init__(self,fitness_func=testfun,dim=1,prangetup=None,constraints=None,opttype=0,expmax=10000):
		self.fitness=fitness_func	#function variable i.e. variable refrencing to a function
		self.inputdim=dim	
		self.rangetup=prangetup			#a tuple of form ( start, end)
		self.constraints=constraints #type yet to be decided
		self.opttype=opttype	#i.e. minimize or maximize 
		self.expmax=expmax 		#i.e. expected maximum to be used in find expectation
	def find_expectation(self,xarr):
		if self.opttype==1:
			return find_fitness(xarr)
		else:
			return self.expmax+100-self.find_fitness(xarr)

	def find_fitness(self,xarr):#xarr  is  vector   type
		return self.fitness(xarr)







	