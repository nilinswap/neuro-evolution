
#nsga.py

import numpy as np 
import population
def fir_dominates_sec(popul,i,j):
	n=popul.objarr.shape[1]
	count=0
	for col in range(n):

		if popul.objarr[i][col]>popul.objarr[j][col]:
			return False
		if popul.objarr[i][col]==popul.objarr[j][col]:
			count+=1
	if count==n:
		return False
	return True
def find_distance(a,b):
	return sum((a-b)**2)
def find_share_fac(popul,newfront,ind1,ind2,sigma_share):
	
	if find_distance(popul.poparr[ind1],popul.poparr[ind2])>sigma_share:
		return 0
	else:
		return 1-find_distance(popul.poparr[ind1],popul.poparr[ind2])**2/sigma_share**2
def fit_share_func(popul,newfront,ind,sigma_share):#returns a list
	lis=[find_share_fac(popul,newfront,frontind,ind,sigma_share) for frontind in newfront if frontind!=ind]
	return lis
def return_fitarr(popul,sigma_share=2):

	restset=set(list(np.arange(popul.size)))
	dum_fitness=10**10
	fitarr=np.zeros((popul.size,))
	fitarr+=np.inf# here could be an error
	while restset:
		newfront=set([])
		for i in restset:
			flag=0
			for j in restset:
				if  fir_dominates_sec(popul,j,i):
					flag=1
					break
			if not flag:
				newfront.add(i)
		print(newfront)
		restset=restset-newfront
		for ind in newfront:
			sum_share=sum(fit_share_func(popul,newfront,ind,sigma_share))
			if sum_share==0:
				fitarr[ind]=dum_fitness	#here could be an error
			else:
				fitarr[ind]=dum_fitness/sum_share
			print(ind,fitarr[ind],sum_share)	
				
		dum_fitness=0.99*min(fitarr)

	return -fitarr