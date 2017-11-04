#gmachine.py
import matplotlib.pyplot as plt
import misc
import problem
import population
import numpy as np
import problem1_obj as po
import pylab as pl
import matplotlib.pyplot as plt
def smallstepchange(newborn,fac=3,rate=0.1):
		D=len(newborn)

		for k in range(D):
		    if np.random.uniform(0,1) < rate:
		        newborn[k]=newborn[k]+float(np.random.randn(1,1)/3)
		return newborn
		


		
def main():
	mini=np.inf
	popsize=1100
	prob = problem.Problem(prob_func=po.problem1_func, randomvec_func=po.problem1_randomvec_func,
						   constraints=po.constraints, dim=3, prangetup=(-100, 100))

	popu = population.Population(prob, size=popsize)
	popu.randominit()
	genlim=200
	#print(popu.poparr)
	#print("thing starts here")
	newsel=misc.Selection(1)
	newcros=misc.Crossover(rate=0.8)
	newmuta=misc.Mutation(rate=0.1)
	mrate=0.2
	newterm=misc.Termination(0)
	#print(popu.avg_fitness())


	for i in range(genlim):
		lis=[]
		if (i==200):
			print("here it is-----------------------------------------")
		if (i==5000):
			print("here it is 500 ------------------------------------")
			mrate=0.02
			newmuta=misc.Mutation(rate=0.01)
		
		#popu.set_fitarr()
		#minic=min(popu.fitarr)



		for j in range(popu.size):
			if popu.poparr[j][0]<0:
				print("reporting issue -ve")
				exit()
		for tup in newsel.select_parent(popu):

			child1,child2=newcros.do_crossover(tup)
			if child1[0]<0:
				print ("anarth h here")
			child1=newmuta.mutate(child1,prob.constraints)
			child2=newmuta.mutate(child2,prob.constraints)
			for p in range(popu.size):
				if popu.poparr[p][0] < 0:
					print("reporting issue -ve")
					exit()
			lis.append(child1)
			lis.append(child2)
		del(popu)
		#print("below here")

		arr=np.array(lis)
		popu=population.Population(prob,poparr=arr,size=popsize)
		#print("below here")
		popu.set_fitarr()
		if np.all(popu.poparr==popu.poparr[0] ):
			break									#these two conditionals have pen-ultimate IMPORTANCE, as my normalization fails heavily if all are same
		if np.all(popu.fitarr==popu.fitarr[0]):
			break
		#print(popu.avg_fitness(),i)
		#print(mini,list(mininp))
		print(i,genlim)
		if  newterm.terminate(popul=popu,generationnum=i,generationlim=genlim):
			print("breaking bad")
			break
		
	print(popu.poparr)

	#print(popu.avg_fitness())

main()
