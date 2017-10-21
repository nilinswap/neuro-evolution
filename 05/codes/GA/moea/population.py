#poplutaion.py
import problem
import misc
import numpy
import nsga
import problem1_obj as po			
class Population:
	def __init__(self,prob,poparr=numpy.array([]),size=0,stadym=0):
		self.poparr=poparr	# numpy ndarray of tuples of dim
		self.size=size
		self.prob=prob
		self.stadym=stadym	#that size is static or dynamic
		self.objarr=numpy.array([])
		self.fitarr=numpy.array([])
			
	def set_objarr(self):
		
			self.objarr=numpy.array(list(map(self.prob.find_obj,self.poparr)))       #2d list.
			print("objarr",self.objarr)
	"""
	def find_expecarr(self):
		
		if not len(self.fitarr):
			self.set_fitarr()
		p=self.fitarr
		

		mini=min(p)
		maxi=max(p)
		numpy.seterr(divide='ignore', invalid='ignore') 	#here  could be an error
		newp=(p-mini)/(maxi-mini)
		
		#now it is normalized
		if self.prob.opttype==1:
			return newp
		elif self.prob.opttype==0:
			return 1-newp
	"""
	def randominit(self):
		self.poparr=numpy.array([self.prob.return_random() for i in range(self.size)])
	
	def set_fitarr(self):
		if not len(self.objarr):
			self.set_objarr()
		self.fitarr=nsga.return_fitarr(self)
		print(self.fitarr)

def main():
	prob=problem.Problem(prob_func=po.problem1_func,randomvec_func=po.problem1_randomvec_func,constraints=po.constraints,dim=5,prangetup=(-100,100))
	
	popu=Population(prob,size=10)

	popu.randominit()
	
	print("problem starts here")
	newsel=misc.Selection(1)
	newcros=misc.Crossover()
	newmuta=misc.Mutation()
	"""par1,par2=popu.poparr[0],popu.poparr[1]
	print(par1,par2)
	for i in range(100):
		child1,child2=newcros.do_crossover((par1,par2))
		child1=newmuta.mutate(child1,prob.constraints,switch=True,iteri=i)
		child2=newmuta.mutate(child2,prob.constraints,iteri=i)
		par1,par2=child1,child2
		print(child1,child2)
	"""
	count=0
	for i in newsel.select_parent(popu):
		
		child1,child2=newcros.do_crossover(i)

		child1=newmuta.mutate(child1,prob.constraints,switch=True,iteri=count)
		child2=newmuta.mutate(child2,prob.constraints,iteri=count)
		print(child1,child2)
		count+=1
	

	#print(popu.prob.expmax)
if __name__ == '__main__':
	main()


