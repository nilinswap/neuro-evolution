#gmachine.py
import misc
import problem
import population
import numpy as np
def mutachange(nowavg,prevavg,prevtoprevavg,mrate,step=0.01):# so far programmed for minimization only
	if (prevavg-prevtoprevavg)>step and (nowavg-prevavg)>step:
		print("to next level")
		mrate=mrate/10
		newmuta=misc.Mutation(rate=mrate)

		prevtoprevavg=2
		prevavg=1
		nowavg=0
		return (newmuta,nowavg,prevavg,prevtoprevavg,mrate)
	return None
def kickchange(popu,nowavg,prevavg,prevtoprevavg,kickrate=0.05,step=0.01):
	if np.random.rand()<kickrate and abs(prevtoprevavg-prevavg)<step and abs(nowavg-prevavg)<step:
			print("kick change ***************************************************************")
			popu.poparr=np.array([i+np.random.rand()/1000 for i in popu.poparr])
def main():
	popsize=1100
	
	mrate=0.1
	minavg=np.inf
	prob=problem.Problem(fitness_func=problem.rastrigin,dim=10,prangetup=(-100,100))
	popu=population.Population(prob,size=popsize)
	popu.randominit()
	genlim=20000
	print(popu.poparr)
	print("thing starts here")
	newsel=misc.Selection(1)
	newcros=misc.Crossover(rate=0.9)
	newterm=misc.Termination(1)	
	prevtoprevavg=2
	prevavg=1
	nowavg=0
	flag=0
	newmuta=misc.Mutation(rate=mrate)

	print(mrate,"starts")
	
	
	
	print(popu.avg_fitness())
	for i in range(genlim):
		lis=[]
		if (i==200):
			print("here it is-----------------------------------------")
		if (i==500):
			print("here it is 500 ------------------------------------")
		tup=mutachange(nowavg,prevavg,prevtoprevavg,mrate,step=0.001)
		if tup:
			(newmuta,nowavg,prevavg,prevtoprevavg,mrate)=tup
		kickchange(popu,nowavg,prevavg,prevtoprevavg,kickrate=0.05,step=0.001)	#this changes population array
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
			flag=1
			break
		prevtoprevavg=prevavg
		prevavg=nowavg
		nowavg=popu.avg_fitness()
		


		
		
	print(popu.poparr)

	print(popu.avg_fitness())

main()
