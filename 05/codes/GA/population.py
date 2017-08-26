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
		p=self.fitarr
		mini=min(p)
		maxi=max(p)

		newp=(p-mini)/(maxi-mini)
		if self.prob.opttype==1:
			return newp
		elif self.prob.opttype==0:
			return 1-newp
		
	def avg_fitness(self):
		self.set_fitarr()
		return numpy.mean(self.fitarr)
	

	def randominit(self):
		self.poparr=numpy.random.uniform(self.prob.rangetup[0],self.prob.rangetup[1],(self.size,self.prob.inputdim))


def main():
	prob=problem.Problem(dim=3,prangetup=(-3,3))
	popu=Population(prob,size=10)

	popu.randominit()
	print(popu.poparr)
	print(popu.avg_fitness())
	print(popu.fitarr)
	print(popu.find_expecarr())
	print("problem starts here")
	newsel=misc.Selection()
	newcros=misc.Crossover()
	newmuta=misc.Mutation()

	for i in newsel.select_parent(popu):
		print("before",i[0],i[1])
		child1,child2=newcros.do_crossover(i)

		child1=newmuta.mutate(child1)
		child2=newmuta.mutate(child2)
		print(child1,child2)



	#print(popu.prob.expmax)
if __name__ == '__main__':
	main()


