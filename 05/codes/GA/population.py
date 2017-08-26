#poplutaion.py
import problem
import misc
import numpy
class Population:
	def __init__(self,prob,poparr=numpy.array([]),size=0,stadym=0):
		self.poparr=poparr	# numpy ndarray of tuples of dim
		self.size=size
		self.prob=prob
		self.stadym=stadym	#that size is static or dynamic
		self.fitarr=numpy.array([])
	def set_fitarr(self):
		self.fitarr=numpy.array(list(map(self.prob.find_fitness,self.poparr)))
	def find_expecarr(self):
		return  numpy.array(list(map(self.prob.find_expectation,self.poparr)))
	def avg_fitness(self):
		self.set_fitarr()
		return numpy.mean(self.fitarr)
	def make_map_from_value_to_input(self):
		if len(self.fitarr):
			bip=numpy.array([round(i,6) for i in self.fitarr])
			self.map={p:q for p,q in zip(bip,self.poparr)}	#round helps in mapping; why 6 is just a good question.


	def randominit(self):
		self.poparr=numpy.random.uniform(self.prob.rangetup[0],self.prob.rangetup[1],(self.size,self.prob.inputdim))

def main():
	prob=problem.Problem(dim=3,prangetup=(-3,3))
	popu=Population(prob,size=10)

	popu.randominit()
	print(popu.poparr)
	print(popu.avg_fitness())
	print(popu.fitarr)
	popu.make_map_from_value_to_input()
	print(popu.map)
	print(popu.find_expecarr())
	print("problem starts here")
	newsel=misc.Selection()
	for i in newsel.select_parent(popu):
		print(i)

	#print(popu.prob.expmax)
if __name__ == '__main__':
	main()


