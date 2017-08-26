#misc.py



class Selection:
	def __init__(self,typeh=0):
		self.type=typeh
		
	def select_parent(self):
		if self.type==0:
			return WeihRoulWheel(popul)	#takes in population, returns a tuple of two vectors
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
				return middlepoint(parent_tup)	#returns a tuple of children(vectors)

			elif self.type==1:
				return singlepoint()	
			elif self.type==2:
				return doublepoint()
			elif self.type==3:
				#use alitism
				return None

class Mutation:
	def __init__(self,typeh=0,rate=0.1,stadym=0):
		self.type=typeh
		self.rate=rate
		self.stadym=stadym

	def do_crossover(self,newborn):
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

