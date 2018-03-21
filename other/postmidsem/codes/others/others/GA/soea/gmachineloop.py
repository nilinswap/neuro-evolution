#gmachine.py
import misc
import problem
import population
import numpy as np
def main():
	popsize=1100
	
	mrate=0.1
	minavg=np.inf
	while mrate>0.0005:
		prob=problem.Problem(fitness_func=problem.rastrigin,dim=10,prangetup=(-100,100))
		popu=population.Population(prob,size=popsize)
		popu.randominit()
		genlim=2000
		print(popu.poparr)
		print("thing starts here")
		newsel=misc.Selection()
		newcros=misc.Crossover(rate=0.9)
		newterm=misc.Termination(1)
		print(mrate,"starts")
		newmuta=misc.Mutation(rate=mrate)
		
		print(popu.avg_fitness())
		for i in range(genlim):
			lis=[]
			if (i==200):
				print("here it is-----------------------------------------")
			if (i==500):
				print("here it is 500 ------------------------------------")
			if np.random.rand()<0.05:
				print("been here***************************************************************")
				popu.poparr=np.array([i+np.random.rand() for i in popu.poparr])

			for tup in newsel.select_parent(popu):
				child1,child2=newcros.do_crossover(tup)
				child1=newmuta.mutate(child1,prob.rangetup, switch=1,iteri=i,switchiter=100,factup=(100,1000))
				child2=newmuta.mutate(child2,prob.rangetup, switch=1,iteri=i,switchiter=100,factup=(100,1000))
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
			print(popu.avg_fitness(),mrate,i)
			if  newterm.terminate(popul=popu,iteri=i,lim=500):
				print("breaking bad")
				print("for mrate ",mrate," last gen avg is", popu.avg_fitness())
				break
		mrate/=10
		
	print(popu.poparr)

	print(popu.avg_fitness())

main()
