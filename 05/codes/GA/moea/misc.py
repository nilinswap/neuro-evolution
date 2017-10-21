#misc.py
import numpy as np
import nsga
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

def gen_randuniform(low,high,size):
	while True:
		try:
			r=np.random.uniform(low,high,size)
			return r
		except:
			try:
				r=np.random.normal(low,high,size)
				return r
			except:
				continue
			
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
def NewRoulWheel(arr):#this uses tournament inside it, which sounds crazy but accuracy is imporved for inputs we tested on.
	sumarr=[0]

	for i in range(len(arr)):
		sumarr.append(sumarr[i]+arr[i])
	n=len(arr)//2
	global state
	np.random.seed(state)#this was important so that the random stream does not run out .... may be 
	#RANDOM COULD BE A PROBLEM, AND ITS SEEDING. 
	state+=1
	
	for j in range(n):
		r = np.random.uniform(0,sumarr[-1],4)#gen_randuniform(0,sumarr[-1],2)
		chosenind1a=binsear(r[0],sumarr)
		chosenind2a=binsear(r[1],sumarr)
		chosenind1b=binsear(r[2],sumarr)
		chosenind2b=binsear(r[3],sumarr)

		if chosenind1a>chosenind1b:
			chosenind1=chosenind1a
		else:
			chosenind1=chosenind1b

		if chosenind2a>chosenind2b:
			chosenind2=chosenind2a
		else:
			chosenind2=chosenind2b

		yield (chosenind1,chosenind2)
	
def WeighRoulWheel(popul):
	ar=popul.find_expecarr()
	for tup   in   RoulWheel(ar):
			yield popul.poparr[tup[0]],popul.poparr[tup[1]]


def NewRankRoulWheel(popul):
	ar=np.arange(0,popul.size)
	par=list(-popul.fitarr)
	
	listup=list(zip(list(popul.poparr),par))
	listup.sort(key=lambda x: x[1])

	for  tup in NewRoulWheel(ar):
		

		yield listup[tup[0]][0],listup[tup[1]][0]
def RankRoulWheel(popul):
	ar=np.arange(0,popul.size)
	if not len(popul.fitarr):
		popul.set_fitarr()
	par=list(-popul.fitarr)
	
	listup=list(zip(list(popul.poparr),par))
	listup.sort(key=lambda x: x[1])

	for  tup in RoulWheel(ar):
		

		yield listup[tup[0]][0],listup[tup[1]][0]

def middlepoint(parent_tup):
	alpha=np.random.uniform(0,1)
	child1=alpha*parent_tup[0]+(1-alpha)*parent_tup[1]
	child2=alpha*parent_tup[1]+(1-alpha)*parent_tup[0]
	return (child1,child2)
def smallchangeonedir(newborn,constraints,fac=1000):
	p=np.random.normal(-1,1,newborn.shape)
	if not constraints:
		return p
	for i in newborn.shape[0]:
		if i in constraints:
			temp=np.random.normal(-1,1)
			p[i]=temp
			while not constraints[i](p):
				temp=np.random.normal(-1,1)
				p[i]=temp
		else :
			temp=np.random.normal(-1,1)
			p[i]=temp
			while not constraints['rest'](p,i):
				temp=np.random.normal(-1,1)
				p[i]=temp
	return p
def is_satisfied(p,constraints):
	
	if not constraints:
		return True
	for i in range(p.shape[0]):
		if i in constraints:
			if not constraints[i](p):
				return False
		elif not constraints['rest'](p,i):
			return False
	return True
			
				
			
def smallchangealldir(newborn,constraints,fac=1000):
	p=newborn+np.random.normal(-1,1,newborn.shape)/(fac)
	while not is_satisfied(p,constraints):
		p=newborn+np.random.normal(-1,1,newborn.shape)/(fac)
	return p



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
		elif self.type==3:
			return NewRankRoulWheel(popul)

class Crossover:
	def __init__(self,typeh=0,rate=1,stadym=0):
		self.type=typeh
		self.rate=rate
		self.stadym=stadym

	def do_crossover(self,parent_tup):
		if np.random.rand()<self.rate:

			if self.type==0:
				return middlepoint(parent_tup)	#returns a tuple of children(vectors)

			elif self.type==1:
				return singlepoint()	
			elif self.type==2:
				return 
			elif self.type==3:
				#use alitism
				return None

		return parent_tup

class Mutation:
	def __init__(self,typeh=0,rate=0.1,stadym=0):
		self.type=typeh
		self.rate=rate
		self.stadym=stadym

	def mutate(self,newborn,constraints,switch=None,iteri=None,switchiter=100,factup=(100,1000)):

		if not switch:
			if np.random.rand()<self.rate:

				if self.type==0:
					return smallchangealldir(newborn,constraints)	#returns a children (vector)

				elif self.type==1:
					return None
			else:
				return newborn
		else:
			if np.random.rand()<self.rate:
				if  iteri>switchiter:
					if self.type==0:
						return smallchangealldir(newborn,constraints)	#returns a children (vector)

					elif self.type==1:
						return None
				else:
					if self.type==0:
						return smallchangealldir(newborn,constraints)	#returns a children (vector)

					elif self.type==1:
						return None
			else:
				return newborn
class Termination:
	def __init__(self,typeh=0):
		self.type=typeh
	def terminate(self,generationnum=None,generationlim=100,popul=None,iteri=None,lim=500):
		if  self.type==0 :
			if generationnum>generationlim:
				return  1
			else:
				return  0

		elif   self.type==1:
			if 	iteri>lim:
				
				from collections import Counter
				roundar=np.round(popul.poparr,4)
				lis=roundar.tolist()

				lis=[tuple(i) for i in lis]
				counts=Counter(lis)
				counts=dict(counts)
				lis=list(counts.items())
				lis.sort(key= lambda x: x[1],reverse=True)
				if lis[0][1]>popul.size//2:
					print( lis[0][0])
					return 1
			else:
				return 0

			

