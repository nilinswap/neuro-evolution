import numpy as np 
import theano
import theano.tensor as T 
import pimadataf
rng=np.random
n_in=8
n_out=1
n_hid=120

W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_hid)),
                    high=np.sqrt(6. / (n_in + n_hid)),
                    size=(n_in, n_hid)
                ),
                dtype=theano.config.floatX
            )
            

w1 = theano.shared(value=W_values, name='w1', borrow=True)
b_values = np.zeros((n_hid,), dtype=theano.config.floatX)
b1 = theano.shared(value=b_values, name='b1', borrow=True)

w2=theano.shared(
            value=np.zeros(
                (n_hid, n_out),
                dtype=theano.config.floatX
            ),
            name='w2',
            borrow=True
        )
b2=theano.shared(
            value=np.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b2',
            borrow=True
        )

lis=pimadataf.give_datainshared()
	
rest_setx,rest_sety=lis[0]#tuple of two shared variable of array
test_setx,test_sety=lis[1]#tuple of shared variable of array

x=T.matrix('x')
y=T.ivector('y')

lin_midout=T.dot(x,w1)+b1
midout=T.tanh(lin_midout)
params1=[w1,b1]
printmidout=theano.function([x],midout[0])
lin_out=T.dot(midout,w2)+b2
output=T.nnet.sigmoid(lin_out)
printout=theano.function([x],output[0])
params2=[w2,b2]
print("=0000000000000432-932=08999999999999999999999999999999999999999999999999999")
error=0.5*T.mean((y-output.reshape((x.shape[0],)))**2)
finalerror=T.mean(abs(y-(output.reshape((x.shape[0],)))))
fun=theano.function([],rest_sety)
fund=theano.function([],test_sety)
params=params1+params2
learning_rate=0.01
gparams=[T.grad(error,j) for j in params]
updates = [
		    (param, param - learning_rate * gparam)
		    for param, gparam in zip(params, gparams)
		]
print(fun())
train_model=theano.function([],error,updates=updates,givens={x:rest_setx,y:rest_sety})
fun1=theano.function([x],output.reshape((x.shape[0],)))

print(fun())

for i in range(1000):
	#p=train_model(rest_setx.get_value(),fun())
	p=train_model()
	#print(fun1(rest_setx.get_value()))
	#print(fun()-fun1(rest_setx.get_value()))
	heyout=fun1(rest_setx.get_value())
	print(np.where(heyout>0.5,1,0))
	#print(w1.get_value(),w2.get_value())
	#print("mid",printmidout(rest_setx.get_value()))
	#print("out",printout(rest_setx.get_value()))
	print(p)
test_model=theano.function([],[error,finalerror],givens={x:test_setx,y:test_sety})
heyout=fun1(test_setx.get_value())
#print(np.where(heyout>0.5,1,0))
print("here",test_model()[0],np.mean(abs(fund()-np.where(heyout>0.5,1,0))))

