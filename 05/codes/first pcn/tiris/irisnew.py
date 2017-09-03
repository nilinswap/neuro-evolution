import pcn

import numpy as np
import pylab as pl
def myrange(start,end,step):
	i=start
	while i+step < end:
		i+=step
		yield i



def convert_iris():
	
	fileo=open("/home/swapnil/forgit/neuro-evolution/05/dataset/iris/iris.data","r+")
	stlis=fileo.readlines()
	stlis=[i.split(',') for i in stlis]
	fileo.close()
	

	for i in range(len(stlis)-1):
		if stlis[i][4]=='Iris-setosa\n':
			stlis[i][4]='1,0,0\n'
		elif stlis[i][4]=='Iris-versicolor\n':
			stlis[i][4]='0,1,0\n'
		elif stlis[i][4]=='Iris-virginica\n':
			stlis[i][4]='0,0,1\n'#using 1 of N encoding
	stlis=[','.join(i) for i in stlis]
	st=''.join(stlis)
	fileob=open("/home/swapnil/forgit/neuro-evolution/05/dataset/iris/newiris.data","w")
	fileob.write(st)
	fileob.close()
	return len(stlis)-1
def standardize_dataset(traindata, means, stdevs):
    for row in traindata:
        for i in range(len(row)):

            row[i] = (row[i] - means[i])
            if stdevs[i]:
                row[i]/=stdevs[i]

def give_data():
	#1. make iris.data in usable form
	#2. make input set and output set out of it
	#3. make setpool out of the dataset
	#4. make pcn and train it
	#5. test on validation and testing set



	convert_iris()
	irisdata=np.loadtxt("/home/swapnil/forgit/neuro-evolution/05/dataset/iris/newiris.data", delimiter=',')
	order=np.arange(np.shape(irisdata)[0])
	np.random.shuffle(order)
	irisdata = irisdata[order,:]
	nin =4;
	nout=3;
	irisdata=irisdata.astype(float)
	traindata=irisdata
	means= traindata.mean(axis=0)

	stdevs=np.std(traindata,axis=0)
	# standardize dataset
	standardize_dataset(traindata[:,:4],means,stdevs)
	rest_setx=irisdata[:120,:4]#tuple of two shared variable of array
	rest_sety=irisdata[:120,4:]
	test_setx=irisdata[120:,:4]
	test_sety=irisdata[120:,4:]
	return ((rest_setx,rest_sety),(test_setx,test_sety))

def find_fitness(rest_setx,rest_sety,weightarr):
	
	
	rows=np.shape(rest_setx)[1]+1	#for bias
	cols=np.shape(rest_sety)[1]
	
	#weightarr=np.array([[weightarr[i*rows+j] for j in range(cols) ] for i in range(rows) ])
	weightarr=np.reshape(weightarr,(5,3))
	
	net=pcn.pcn(rest_setx,rest_sety,weightarr)
	arr=net.pcnfwd(rest_setx)
	
	er_arr=(1/2)*np.mean((arr-rest_sety)**2)
	
	return (er_arr)
def irismain():
	(rest_setx,rest_sety)=give_data()[0]
	lis=list(np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14]))
	print(find_fitness(rest_setx,rest_sety,lis))
irismain()