import theano
import theano.tensor as T
from theano.tensor.nnet import nnet
import numpy as np
rng=np.random
def tconfmat(y,predictions,outputdim):# both y and predictions are vector of P
		
		# Add the inputs that match the bias node
		
	
		nClasses = outputdim

		if nClasses==1:
			nClasses = 2
			outputs = np.where(predictions>0,1,0)
			targets=y
		else:
			# 1-of-N encoding
			outputs = np.argmax(predictions,1)
			targets = np.argmax(y,1)

		cm = np.zeros((nClasses,nClasses))
		for i in range(nClasses):
			for j in range(nClasses):
				cm[i,j] = np.sum(np.where(outputs==i,1,0)*np.where(targets==j,1,0))

		print (cm)
		err= 1-(np.trace(cm)/np.sum(cm)) 
		print(err)
		return err 
def feednextbatch(tup):
	i=0
	n=len(tup[0])
	print("here",n)
	while i<n:
		yield (tup[0][i:i+1000],tup[1][i:i+1000])
		i+=1000

from mnist import MNIST
mndata = MNIST('./dataset')
traintup=mndata.load_training()



traindata=traintup[0]#60000 X 784
traindata=list(traindata)
traindata=np.array(traindata)
trainlabel=traintup[1] #vector of 60000
trainlabel=list(trainlabel)
trainlabel=np.array(trainlabel)
newtraintup=(traindata,trainlabel)
"""
traindata=traindata[:3000]
trainlabel=trainlabel[:3000]

testtup=mndata.load_testing()

testdata=testtup[0]#10000 X 784
testlabel=testtup[1] #vector of 10000
"""
print(type(traindata[:3]))
print(type(newtraintup[0]),np.shape(newtraintup[0]))
print(type(newtraintup[1]),np.shape(newtraintup[1]))


inputdim=784
outputdim=10


training_steps=500

x=T.dmatrix("x") # P X i
tar=T.ivector("tar")#vector of  P, this could be P X N, just confmat had to be different 


w=theano.shared(rng.randn(inputdim,outputdim),name="w")# i X N
b=theano.shared(rng.randn(outputdim),name="b")#P X N

y = T.nnet.softmax(T.dot(x,w) + b)	
cost = T.nnet.categorical_crossentropy(y, tar).mean()#check here
gw, gb = T.grad(cost, [w, b])
eta=0.2
train = theano.function(
          inputs=[x,tar],
          outputs=[y.argmax(axis=1)-tar, cost],
          updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)))
predict = theano.function(inputs=[x], outputs=y.argmax(axis=1))

for i in range(training_steps):
	#for tup in feednextbatch(newtraintup):
	xv=newtraintup[0]
	tarv=newtraintup[1]
	tupu=train(xv,tarv)
	lis=[]
	for i in range(len(tupu[0])):
		if tupu[0][i]!=0:
			lis.append(1)
		else:
			lis.append(0)

	print(np.array(lis).mean())
	
	"""temp=T.dot(x,w)+b #P X N
	dotifun=theano.function([x],temp)
	befactiv=dotifun(xv)
	
	temp=T.matrix()
	g=T.nnet.sigmoid(temp)
	sig=theano.function([temp],g)
	z=sig(befactiv)"""

	
	
"""
if sigac==0:
	sigactiv=nnet.sigmoid(befactiv)#P X N
elif sigac==1:
	sigactiv=nnet.ultra_fast_sigmoid(befactiv)
elif sigac==2:
	sigactiv=nnet.hard_sigmoid(befactiv)

activation=nnet.softmax(befactiv)
#activation=nnet.softmax(sigactiv) #P X N
predictions=activation.argmax(axis=1)# vector of size P
if errtype==0:
	
	#xent=nnet.categorical_crossentropy(predictions,y)
	xent = -y * T.log(predictions) - (1-y) * T.log(1-predictions)
	cost = xent.mean() #+ 0.01 * (w ** 2).sum()# The cost to minimize
	gw, gb = T.grad(cost, [w, b])#VERY important
	
	
else:
	pass

train = theano.function(
          inputs=[x,y],
          outputs=[predictions-y, xent],
          updates=((w, w - eta * gw), (b, b - eta * gb)))
feedforward = theano.function(inputs=[x], outputs=predictions)
print(np.shape(traindata),np.shape(trainlabel))
train(traindata,trainlabel)
print(sigactiv[:3])


for i in range(training_steps):
    pred, err = train(traindata,trainlabel)
    
print("Final model:")
#print(w.get_value())
#print(b.get_value())

print("also")
print(pred,err)
"""


