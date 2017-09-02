#gmachine.py
import misc
import problem
import population
import numpy as np

def smallstepchange(newborn,fac=3,rate=0.1):
		D=len(newborn)

		for k in range(D):
		    if np.random.uniform(0,1) < rate:
		        newborn[k]=newborn[k]+float(np.random.randn(1,1)/3)
		return newborn
		


		
def main():
	mini=np.inf
	popsize=1100
	prob=problem.Problem(fitness_func=problem.katsuura,dim=10,prangetup=(-100,100))
	popu=population.Population(prob,size=popsize)
	popu.randominit()
	genlim=20000
	print(popu.poparr)
	print("thing starts here")
	newsel=misc.Selection(3)
	newcros=misc.Crossover(rate=0.8)
	#newmuta=misc.Mutation(rate=0.1)
	mrate=0.2
	newterm=misc.Termination(0)
	print(popu.avg_fitness())
	for i in range(genlim):
		lis=[]
		if (i==200):
			print("here it is-----------------------------------------")
		if (i==5000):
			print("here it is 500 ------------------------------------")
			mrate=0.02
			#newmuta=misc.Mutation(rate=0.01)
		
		popu.set_fitarr()
		minic=min(popu.fitarr)

		if mini>minic:
			mini=minic
			mininp=popu.poparr[list(popu.fitarr).index(minic)]


		for tup in newsel.select_parent(popu):
			child1,child2=newcros.do_crossover(tup)
			child1=smallstepchange(child1,fac=3,rate=mrate)
			child2=smallstepchange(child2,fac=3,rate=mrate)
			lis.append(child1)
			lis.append(child2)
		del(popu)
		

		arr=np.array(lis)
		popu=population.Population(prob,poparr=arr,size=popsize)
		popu.set_fitarr()
		if np.all(popu.poparr==popu.poparr[0] ):
			break									#these two conditionals have pen-ultimate IMPORTANCE, as my normalization fails heavily if all are same
		if np.all(popu.fitarr==popu.fitarr[0]):
			break
		print(popu.avg_fitness(),i)
		print(mini,list(mininp))

		if  newterm.terminate(popul=popu,generationnum=i,generationlim=genlim):
			print("breaking bad")
			break
		
	print(popu.poparr)

	print(popu.avg_fitness())

main()
