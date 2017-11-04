import numpy as np
def x0fun(x):
	return x[0]>=0 and x[0]<=1
def restfun(x,i):
	return x[i]>=-1 and x[i]<=1
constraints={0:x0fun,'rest':restfun}

def problem1_obj1(x):
	n=x.shape[0]
	lis=list(np.arange(2,n+1))
	J1=[item for item in lis if  item%2]
	J1len=len(J1)
	newlis=[(x[item-1]-np.sin(6*x[0]*np.pi+np.pi*item/n))**2 for item in J1]
	return x[0]+(2/J1len)*sum(newlis)


def problem1_obj2(x):
	n=x.shape[0]
	lis=list(np.arange(2,n+1))
	J2=[item for item in lis if  item%2==0]
	J2len=len(J2)
	newlis=[(x[item-1]-np.sin(6*x[0]*np.pi+np.pi*item/n))**2 for item in J2]
	if x[0]<0:

			print("anarth")
	return 1-np.sqrt(x[0])+(2/J2len)*sum(newlis)
def problem1_func(x):
	return [problem1_obj1(x),problem1_obj2(x)]
def problem1_randomvec_func(n):
	lis=[]
	temp=np.random.random()
	lis.append(temp)
	for i in range(1,n):
		lis.append(np.random.uniform(-1,1))
	return lis
