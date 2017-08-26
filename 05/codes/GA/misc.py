#misc.py
import numpy as np
def binsear(p,arr):
	
	low=0
	high=len(arr)-1
	while (high-low)>1:
		mid=arr[(low+high)//2]
		if mid>p:
			high=(low+high)//2
		else:
			low=(low+high)//2
	return arr[high]-arr[low]



def RoulWheel(arr):
	sumarr=[0]
	for i in range(len(arr)):
		sumarr.append(sumarr[i]+arr[i])
	n=len(arr)//2
	for j in range(n):
		r = np.random.uniform(0, sumarr[-1],2)
		chosen1v=binsear(r[0],sumarr)
		chosen2v=binsear(r[1],sumarr)
		yield (chosen1v,chosen2v)
def WeighRoulWheel(popul):
	ar=popul.find_expecarr()
	for tup   in   RoulWheel(ar):
		if popul.prob.opttype==1:
			yield (popul.map[tup[0]],popul.map[tup[1]])

		elif popul.prob.opttype==0:
			h1=-(tup[0]-popul.prob.expmax-100)
			h2=-(tup[1]-popul.prob.expmax-100)
			yield (popul.map[round(h1,6)],popul.map[round(h2,6)])
def RankRoulWheel(popul):
	ar=np.arange(1,popul.size+1)
	par=popul.fitarr
	if popul.prob.opttype==0:
		par[::-1].sort()
	elif popul.prob.opttype==1:
		par.sort()
	for  tup in RoulWheel(ar):
		yield (popul.map[par[tup[0]-1]],popul.map[par[tup[1]-1]])



class Selection:
	def __init__(self,typeh=0):
		self.type=typeh
		
	def select_parent(self,popul):
		if self.type==0:
			return WeighRoulWheel(popul)	#takes in population, returns a tuple of two vectors
		elif self.type==1:
			return RankRoulWheel(popul)
		elif self.type==2:
			#use alitism
			return None

class Crossover:
	def __init__(self,typeh=0,rate=1,stadym=0):
		self.type=typeh
		self.rate=rate
		self.stadym=stadym

	def do_crossover(self,parent_tup):
		if np.random.rand()<self.rate:

			if self.type==0:
				return doublepoint(parent_tup)	#returns a tuple of children(vectors)

			elif self.type==1:
				return singlepoint()	
			elif self.type==2:
				return 
			elif self.type==3:
				#use alitism
				return None

class Mutation:
	def __init__(self,typeh=0,rate=0.1,stadym=0):
		self.type=typeh
		self.rate=rate
		self.stadym=stadym

	def do_mutation(self,newborn):
		if np.random.rand()<self.rate:

			if self.type==0:
				return smallchange(newborn)	#returns a children (vector)

			elif self.type==1:
				return None
class Termination:
	def __init__(self,typeh=0):
		self.type=typeh
	def terminate(self,generationnum=None,generationlim=100):
		if  self.type==0:
			if generationnum>generationlim:
				return  1
			else:
				return  0

		elif   self.type==1:

			pass

