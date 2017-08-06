
import pcn
import trainedpcn
import numpy as np
import pylab as pl
def myrange(start,end,step):
	i=start
	while i+step < end:
		i+=step
		yield i

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
			yield returntup

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

def irismain():
	#1. make iris.data in usable form
	#2. make input set and output set out of it
	#3. make setpool out of the dataset
	#4. make pcn and train it
	#5. test on validation and testing set



	convert_iris()
	irisdata=np.loadtxt("/home/swapnil/forgit/neuro-evolution/05/dataset/iris/newiris.data", delimiter=',')
	
	nin=4# for four features of iris
	nout=3# for 3 sets of iris flowers
	minerr=minetaerr=miniterarr=10000000
	switch=2
	etalis=[]
	valerrlis=[]
	niterationslis=[]
	eta=2.5
	niterations=100
	if switch==1:
		for eta in myrange(0.05,0.31,0.0005):
			minetaerr=10000000
			etalis.append(eta)
			for tupoftup in nextpartition(irisdata,nin,nout):
				train,traintarget=tupoftup[0]
				valid,validtarget=tupoftup[1]#each row of setpool is input and their targets so we need to seperate them
				test,testtarget=tupoftup[2]

				#np.concatenate((train,valid),axis=0)
				#np.concatenate((traintarget,validtarget),axis=0)
				#valid is of no use on perceptron because perceptron can not overfit!! and neither is early-stopping.
				net=pcn.pcn(train,traintarget)
				
				net.pcntrain(train,traintarget,eta,niterations)
				print("below")
				err=net.confmat(valid,validtarget)
				print("\n")
				minetaerr=min(minetaerr,err)
				if minerr>err:
					minerr=err
					bestnet=trainedpcn.trainedpcn(net,test,testtarget,err,eta,niterations)
			
			valerrlis.append(minetaerr)#notice this minetaerr is error for each eta 
			# and minerr is the minimum for all minetaerr that is over all eta
		print("\n best network with eta is attained")
		leasterr=bestnet.test()
		print("changing eta, error on test is %f while on valid  is %f" %(bestnet.validmeanerr,leasterr));
		etaarr=np.array(etalis)*np.ones((len(etalis),1))
		valerrarr=np.array(valerrlis)*np.ones((len(valerrlis),1))
		pl.plot(etaarr,valerrarr,'.')
		#pl.plot(x,,'o')
		pl.xlabel('eta')
		pl.ylabel('error')
		
		pl.show()
	elif switch==2:
		for niterations in range(10,1000,10):
			miniteraerr=10000000
			niterationslis.append(niterations)
			for tupoftup in nextpartition(irisdata,nin,nout):
				train,traintarget=tupoftup[0]
				valid,validtarget=tupoftup[1]#each row of setpool is input and their targets so we need to seperate them
				test,testtarget=tupoftup[2]

				#np.concatenate((train,valid),axis=0)
				#np.concatenate((traintarget,validtarget),axis=0)
				#valid is of no use on perceptron because perceptron can not overfit!! and neither is early-stopping.
				net=pcn.pcn(train,traintarget)
				
				net.pcntrain(train,traintarget,eta,niterations)
				print("below")
				err=net.confmat(valid,validtarget)
				print("\n")
				miniteraerr=min(miniteraerr,err)
				if minerr>err:
					minerr=err
					bestnet=trainedpcn.trainedpcn(net,test,testtarget,err,eta,niterations)
			
			valerrlis.append(miniteraerr)
		print("\n best network with eta is attained")
		leasterr=bestnet.test()
		print("changing eta, error on test is %f while on valid  is %f" %(bestnet.validmeanerr,leasterr));
		iterarr=np.array(niterationslis)*np.ones((len(niterationslis),1))
		valerrarr=np.array(valerrlis)*np.ones((len(valerrlis),1))
		pl.plot(iterarr,valerrarr,'.')
		#pl.plot(x,,'o')
		pl.xlabel('iterations')
		pl.ylabel('error')
		
		pl.show()		
			

	
irismain()

