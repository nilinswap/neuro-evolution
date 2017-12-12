import numpy as np 
import theano
import theano.tensor as T
from theano.ifelse import ifelse 
import pimadataf
rng=np.random
n_in=8
n_out=1
n_hid=120
class MLP:


	def __init__(self,n_in,n_out,n_hid,trainx,trainy,testx,testy):

		W_values = np.asarray(
				rng.uniform(
					low=-np.sqrt(6. / (n_in + n_hid)),
					high=np.sqrt(6. / (n_in + n_hid)),
					size=(n_in, n_hid)
				),
				dtype=theano.config.floatX
			)

		self.n_in=n_in
		self.n_out=n_out
		self.n_hid=n_hid
		self.trainx=trainx[:438]
		self.trainy=trainy[:438]
		self.valx = trainx[ 438: ]
		self.valy = trainy[ 438: ]
		self.testx=testx
		self.testy=testy
		self.w1 = theano.shared(value=W_values, name='w1', borrow=True)
		b_values = np.zeros((n_hid,), dtype=theano.config.floatX)
		self.b1 = theano.shared(value=b_values, name='b1', borrow=True)

		self.w2=theano.shared(
					value=np.zeros(
						(n_hid, n_out),
						dtype=theano.config.floatX
					),
					name='w2',
					borrow=True
				)
		self.b2=theano.shared(
					value=np.zeros(
						(n_out,),
						dtype=theano.config.floatX
					),
					name='b2',
					borrow=True
				)
		self.params=[self.w1,self.b1,self.w2,self.b2]
		self.input=self.trainx
		self.vinput = self.valx
		self.tinput=self.testx
		lin_midout=T.dot(self.input,self.w1)+self.b1
		midout=T.nnet.sigmoid(lin_midout)


		lin_out=T.dot(midout,self.w2)+self.b2
		output=T.nnet.sigmoid(lin_out)
		self.output=output

		tlin_midout=T.dot(self.tinput,self.w1)+self.b1
		tmidout=T.nnet.sigmoid(tlin_midout)


		tlin_out=T.dot(tmidout,self.w2)+self.b2
		toutput=T.nnet.sigmoid(tlin_out)
		self.toutput=toutput

		vlin_midout = T.dot(self.vinput, self.w1) + self.b1
		vmidout = T.nnet.sigmoid(vlin_midout)

		vlin_out = T.dot(vmidout, self.w2) + self.b2
		voutput = T.nnet.sigmoid(vlin_out)
		self.voutput = voutput

		self.funb=theano.function([],[self.b1,self.b2])
	def set_weights(self,w1,b1,w2,b2):
		self.w1.set_value(w1)
		self.b1=b1
		self.w2.set_value(w2)
		self.b2=b2
	def set_weights_from_chromosome(self,weightstr):
		new_weightstr=weightstr[1:]
		n_hid=int(weightstr[0])

		w1=new_weightstr[:(self.n_in*self.n_hid)].reshape((self.n_in,self.n_hid))
		b1=new_weightstr[(self.n_in*self.n_hid):(self.n_in*self.n_hid)+self.n_hid].reshape((self.n_hid,))
		w2=new_weightstr[(self.n_in*self.n_hid)+self.n_hid:(self.n_in*self.n_hid)+self.n_hid+(self.n_hid*self.n_out)].reshape((self.n_hid,self.n_out))
		b2=new_weightstr[(self.n_in*self.n_hid)+self.n_hid+(self.n_hid*self.n_out):].reshape((self.n_out,))
		self.set_weights(w1,b1,w2,b2)
	def turn_weights_into_chromosome(self):
		w1,b1,w2,b2=self.get_weights()
		w1=list(w1.reshape(((self.n_in*self.n_hid),)))
		b1,b2=self.funb()#somehow direct access to b1,b2 is barred because of casting, it is not even a shared variable
		b1=list(b1)
		w2=list(w2.reshape(((self.n_hid*self.n_out),)))
		b2=list(b2)

		lis=[float(self.n_hid)]
		lis=lis+w1+b1+w2+b2
		return np.array(lis)



	def get_weights(self):
		tup=(self.w1.get_value(),self.b1,self.w2.get_value(),self.b2)
		return tup

	def cost_func(self,y):
		return 0.5*T.mean((y-self.output.ravel())**2)
	def newcost_func(self,y):
		z=self.output.ravel()
		results,updates=theano.scan(fn=lambda p,q:- p*T.log(q),sequences=[y,z])
		#fun=theano.function([y,z],T.mean(results))


		return T.mean(results)
	def find_error(self,y):
		p=self.toutput.ravel()
		results,updates=theano.scan(fn=lambda x: ifelse(T.lt(x,0.5),0,1),sequences=p)

		return (T.mean(abs(results-y)))

	def find_val_error(self,y):
		p = self.voutput.ravel()
		results, updates = theano.scan(fn=lambda x: ifelse(T.lt(x, 0.5), 0, 1), sequences=p)

		return (T.mean(abs(results - y)))

def main():
	lis=pimadataf.give_datainshared()
		
	rest_setx,rest_sety=lis[0]#tuple of two shared variable of array
	test_setx,test_sety=lis[1]#tuple of shared variable of array
	
	y=T.ivector('y')

	newmlp=MLP(n_in,n_out,n_hid,rest_setx,rest_sety,test_setx,test_sety)

	#error=0.5*T.mean((y-output.reshape((x.shape[0],)))**2)
	#finalerror=T.mean(abs(y-(output.reshape((x.shape[0],)))))
	fun=theano.function([],rest_sety)
	fund=theano.function([],test_sety)
	params=newmlp.params
	cost=newmlp.newcost_func(y)
	learning_rate=0.01
	gparams=[T.grad(cost,j) for j in params]
	updates = [
				(param, param - learning_rate * gparam)
				for param, gparam in zip(params, gparams)
			]

	train_model=theano.function([],cost,updates=updates,givens={y:rest_sety})
	#fun1=theano.function([x],output.reshape((x.shape[0],)))
	test_model=theano.function([],newmlp.find_error(y),givens={y:test_sety})
	print(fun())
	epochs=1000
	#net_obj = Net( inputdim, outputdim, arr_of_net, trainx, trainy, testx, testy, strainx, strainy, stestx, stesty)
	for i in range(1,epochs):
		#p=train_model(rest_setx.get_value(),fun())

		p=train_model()

		if i %100==0:
			st=newmlp.turn_weights_into_chromosome()
			newmlp.set_weights_from_chromosome(st)    
			print("been through here")

		#print(fun1(rest_setx.get_value()))
		#print(fun()-fun1(rest_setx.get_value()))
		#heyout=fun1(rest_setx.get_value())
		#print(np.where(heyout>0.5,1,0))
		#print(w1.get_value(),w2.get_value())
		#print("mid",printmidout(rest_setx.get_value()))
		#print("out",printout(rest_setx.get_value()))
		print(p)
	print("here testing",test_model())
if __name__=='__main__':
	main()