

#iris.py
import os.path
import mlp
import trainedmlp
import numpy as np
import pylab as pl
import sklearn
from sklearn import datasets
def makesetpool(inputs,targets,k):
	if k>np.shape(inputs)[0]:
		print("give k properly")
		return -1
	#print(inputs)
	#print(targets)

	nin=np.shape(inputs)[0]
	nout=np.shape(targets)[0]
	dataset=np.concatenate((inputs,targets),axis=1)
	#print(type(dataset))
	setpool=[]
	#print(nin)
	s=True
	for i in range(nin):
		if i<k:
			setpool.append(np.array([dataset[i]]))
		else:
			#print("here")
			 
			setpool[i%k]=np.concatenate((setpool[i%k],np.array([dataset[i]])),axis=0)
	#print(setpool)
	return setpool	#this is a list of arrays of input clubbed with targets
def nextpartition(dataarr,nin,nout):
	k=len(dataarr)//30#following 60 20 20
	setpool=makesetpool(dataarr[:,:nin],dataarr[:,nin:],k)


	for i in range(len(setpool)-1):#testing on all k-1 partitions one by one
			tsetpool=setpool[:]# a simple assignment would just point to same setpool
			valid=tsetpool[i][:,:nin]#each row of setpool is input and their targets so we need to seperate them
			validtarget=tsetpool[i][:,nin:]
			test=tsetpool[i+1][:,:nin]
			testtarget=tsetpool[i+1][:,nin:]
			del tsetpool[i],tsetpool[i]#I am deleting ith and i+1th item from tsetpool,trickily
			newsetpool=[item.tolist() for item in tsetpool]#item is of array type
			lissum=[]
			del tsetpool
			for item in newsetpool:
				lissum+=item #+ can be easily performed with list types therefore converted array to list in above lines
			arr=np.array(lissum)
			
			train=arr[:,:nin]
			traintarget=arr[:,nin:]
			returntup=((train,traintarget),(valid,validtarget),(test,testtarget))
			yield returntup	#eventually I yielded (*smiling meekly*)

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

def irismain(dataset):
	#1. make iris.data in usable form
	#2. make input set and output set out of it
	#3. make setpool out of the dataset
	#4. make pcn and train it
	#5. test on validation and testing set



	#convert_iris()
	if dataset == 'iris':
		irisdata=np.loadtxt("newiris.data", delimiter=',')

		nin=4# for four features of iris
		nout=3# for 3 sets of iris flowers
		order = np.arange(np.shape(irisdata)[0])
		np.random.shuffle(order)
		irisdata = irisdata[order, :]
		irisdata[:, :4] = irisdata[:, :4] - irisdata[:, :4].mean(axis=0)
		imax = np.concatenate((irisdata.max(axis=0) * np.ones((1, 7)), np.abs(irisdata.min(axis=0)) * np.ones((1, 7))),
							  axis=0).max(axis=0)
		irisdata[:, :4] = irisdata[:, :4] / imax[:4]
	elif dataset == '2moon':
		tupa = datasets.make_moons(240, True, 0.01)
		irisdata = np.concatenate((tupa[0], tupa[1].reshape((tupa[1].shape[0], 1))), axis=1)

		nin = 2
		nout = 1
	minerr=10000000
	lis=[]

	errcal='confmat'
	eta=0.29
	niterations=500
	tlis=[]
	for niterations in range(10,1000,10):
		minitererr=10000000
		flag=0
		for nhidden in range(1,6):# range for number of hidden nodes
			minnhiddenerr=100000000
			for tupoftup in nextpartition(irisdata,nin,nout):
				train,traintarget=tupoftup[0]
				valid,validtarget=tupoftup[1]#each row of setpool is input and their targets so we need to seperate them
				test,testtarget=tupoftup[2]

				#np.concatenate((train,valid),axis=0)
				#np.concatenate((traintarget,validtarget),axis=0)
				#valid is of no use on perceptron because perceptron can not overfit!! and neither is early-stopping.
				net=mlp.mlp(train,traintarget,nhidden,outtype='logistic')

				
				net.mlptrain(train,traintarget,eta,niterations//2)
				validmeanerr=net.earlystopping(train,traintarget,valid,validtarget,eta,niterations//10,errcaltype=errcal)
				print("no. of nodes",nhidden)
				lis.append(validmeanerr)
				if errcal=='squaremean':
					trainmeanerr=net.findmeantrainerr(train,traintarget)
				else:
					trainmeanerr=net.confmat(train,traintarget)
				print("validation error: %f trainerr error:%f"%(validmeanerr,trainmeanerr));
				#y=np.concatenate((x,-np.ones((np.shape(x)[0],1))),axis=1)
				minnhiddenerr=min(minnhiddenerr,validmeanerr)
				if validmeanerr<minerr:#see I can't use equal to here so that this way it will select one with lowest num of nodes
						minerr=validmeanerr
						bestnet=trainedmlp.trainedmlp(net,test,testtarget,trainmeanerr,validmeanerr,nhidden)
			if minnhiddenerr<0.07:
				tempnhidden=nhidden
				flag=1
				break
		if not flag:
			tempnhidden=10
		tlis.append((niterations,tempnhidden))
	
	niterationslis=[i[0] for i in tlis]
	temphiddenlis=[i[1] for i in tlis]
	iterarr=np.array(niterationslis)*np.ones((len(niterationslis),1))
	nhiddenarr=np.array(temphiddenlis)*np.ones((len(temphiddenlis),1))
	pl.plot(iterarr,nhiddenarr,'.')
	#pl.plot(x,,'o')
	pl.xlabel('iter')
	pl.ylabel('nhidden')
	
	pl.show()

	print("\n best network is attained with no. of nodes as ",bestnet.numnodes)
	leasterr = bestnet.test()
	print("error on test is %f while on valid  is %f" %(leasterr,bestnet.validmeanerr));

import sys
choice = sys.argv[1]
if choice in {'iris', '2moon'}:
	irismain(choice)
else:
	print("Error!, next time enter one in {'iris', '2moon'} ")


