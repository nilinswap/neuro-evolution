
import numpy as np
def rastrigin(arr):
	n=len(arr)
	lis=[arr[i-1]**2-10*np.cos(2*np.pi*arr[i-1])+10 for i in range(1,n+1)] 
	return sum(lis)

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



def RankRoulWheel(popul):
	ar=np.arange(0,popul.size)
	par=list(-popul.fitarr)
	
	listup=list(zip(list(popul.poparr),par))
	listup.sort(key=lambda x: x[1])

	for  tup in RoulWheel(ar):
		

		yield listup[tup[0]][0],listup[tup[1]][0]





popsize=150
D=10
mini=np.inf
poparr=np.random.uniform(-100,100,(popsize,D))
crossrate=0.8
for gen in range(100000):
	popval=[rastrigin(ar) for ar in poparr]
	popexpec=-np.array(popval)

	
	minic=min(popval)
	ind=popval.index(min(popval))
	if mini>minic:
		mini=minic
		minval=poparr[ind]

	print("min ",mini, "avg", np.mean(popval), "gen ", gen)
	print(minval)
	sumarr=[0]
	arr=np.arange(0,popsize)
	for i in range(len(arr)):
		sumarr.append(sumarr[i]+arr[i])
	n=len(arr)//2
	global state
	np.random.seed(state)#this was important so that the random stream does not run out .... may be 
	#RANDOM COULD BE A PROBLEM, AND ITS SEEDING. 
	state+=1
	listup=list(zip(list(arr),list(popexpec)))
	listup.sort(key=lambda x : x[1])
	
	newpoparr=[]

	for j in range(n):
		r = np.random.uniform(0,sumarr[-1],4)#gen_randuniform(0,sumarr[-1],2)
		chosenind1a=binsear(r[0],sumarr)
		chosenind2a=binsear(r[1],sumarr)
		chosenind1b=binsear(r[2],sumarr)
		chosenind2b=binsear(r[3],sumarr)

		if listup[chosenind1a][1]>listup[chosenind1b][1]:
			chosenind1=chosenind1a
		else:
			chosenind1=chosenind1b

		if listup[chosenind2a][1]>listup[chosenind2b][1]:
			chosenind2=chosenind2a
		else:
			chosenind2=chosenind2b


		par1=poparr[listup[chosenind1][0]]
		par2=poparr[listup[chosenind2][0]]
		if np.random.uniform(0,1)< crossrate:
		    
		    xD=np.random.uniform(0,1)
		    #print(xD)
		    child1=xD*par1+(1-xD)*par2
		    child2=xD*par2+(1-xD)*par1
		else:
			child1=par1
			child2=par2
		
		if gen < 5000:
		   mutation_rate=0.2
		else:
		    	mutation_rate=0.02
		for k in range(D):
		    if np.random.uniform(0,1) < mutation_rate:
		        child1[k]=child1[k]+float(np.random.randn(1,1)/3)
		for k in range(D):
		    if np.random.uniform(0,1) < mutation_rate:
		        child2[k]=child2[k]+float(np.random.randn(1,1)/3)


		
		"""
		if gen<5000:
			mutarate=0.2
		else:
			mutarate=0.02
		fac=3
		lim=[]
		lim.append([-100,100])
		if np.random.uniform(0,1)<mutarate:
				
																#this is important
				child1=child1+np.random.normal(-1,1,D)/(fac)
				

					
																#this is important
				child2=child2+np.random.normal(-1,1,D)/(fac)
		"""	
		newpoparr.append(child1)
		newpoparr.append(child2)
	del(poparr)
	poparr=newpoparr


