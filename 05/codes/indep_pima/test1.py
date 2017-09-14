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
    def __init__(self,inputx,n_in,n_out,n_hid):

        W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_hid)),
                    high=np.sqrt(6. / (n_in + n_hid)),
                    size=(n_in, n_hid)
                ),
                dtype=theano.config.floatX
            )
            

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
        self.input=inputx
        lin_midout=T.dot(self.input,self.w1)+self.b1
        midout=T.tanh(lin_midout)
        
        printmidout=theano.function([self.input],midout[0])
        lin_out=T.dot(midout,self.w2)+self.b2
        output=T.nnet.sigmoid(lin_out)
        self.output=output
    def set_weights(self,w1,b1,w2,b2):
        self.w1.set_value(w1)
        self.b1=b1
        self.w2.set_value(w2)
        self.b2=b2
    def get_weights(self):
        tup=(self.w1.get_value(),self.b1,self.w2.get_value(),self.b2)
        return tup

    def cost_func(self,y):
        return 0.5*T.mean((y-self.output.reshape((self.input.shape[0],)))**2)
    def find_error(self,y):
        p=self.output.reshape((self.output.shape[0],))
        results,updates=theano.scan(fn=lambda x: ifelse(T.lt(x,0.5),0,1),sequences=p)

        return (T.mean(abs(results-y)))

lis=pimadataf.give_datainshared()
	
rest_setx,rest_sety=lis[0]#tuple of two shared variable of array
test_setx,test_sety=lis[1]#tuple of shared variable of array
x=T.matrix('x')
y=T.ivector('y')

newmlp=MLP(x,n_in,n_out,n_hid)

#error=0.5*T.mean((y-output.reshape((x.shape[0],)))**2)
#finalerror=T.mean(abs(y-(output.reshape((x.shape[0],)))))
fun=theano.function([],rest_sety)
fund=theano.function([],test_sety)
params=newmlp.params
cost=newmlp.cost_func(y)
learning_rate=0.01
gparams=[T.grad(cost,j) for j in params]
updates = [
		    (param, param - learning_rate * gparam)
		    for param, gparam in zip(params, gparams)
		]

train_model=theano.function([],cost,updates=updates,givens={x:rest_setx,y:rest_sety})
#fun1=theano.function([x],output.reshape((x.shape[0],)))
test_model=theano.function([],newmlp.find_error(y),givens={x:test_setx,y:test_sety})
print(fun())
epochs=1000
for i in range(1,epochs):
    #p=train_model(rest_setx.get_value(),fun())

    p=train_model()

    if i %100==0:
        tup=newmlp.get_weights()
        newmlp.set_weights(tup[0],tup[1],tup[2],tup[3])    
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
