import pcn

import numpy as np
import pylab as pl
def myrange(start,end,step):
    i=start
    while i+step < end:
        i+=step
        yield i
def give_data():
    #1. make iris.data in usable form
    #2. make input set and output set out of it
    #3. make setpool out of the dataset
    #4. make pcn and train it
    #5. test on validation and testing set



    
    pimadata=np.loadtxt("pima_dataset.csv", delimiter=',')
    print("Hello")
    np.random.shuffle(pimadata)
    
    nin =8;
    nout=2;
    pimadata=pimadata.astype(float)
    traindata=pimadata
    means= traindata.mean(axis=0)

    stdevs=np.std(traindata,axis=0)
    # standardize dataset
    standardize_dataset(traindata[:,:8],means,stdevs)
    rest_setx=irisdata[:538,:8]#tuple of two shared variable of array
    rest_sety=irisdata[:538,8:]
    test_setx=irisdata[538:,:8]
    test_sety=irisdata[538:,8:]
    print(rest_setx)
    print(rest_sety)
    return ((rest_setx,rest_sety),(test_setx,test_sety))
def find_fitness(rest_setx,rest_sety,weightarr):
    
    
    rows=np.shape(rest_setx)[1]+1   #for bias
    cols=np.shape(rest_sety)[1]
    
    #weightarr=np.array([[weightarr[i*rows+j] for j in range(cols) ] for i in range(rows) ])
    weightarr=np.reshape(weightarr,(9,2))
    
    net=pcn.pcn(rest_setx,rest_sety,weightarr)
    arr=net.pcnfwd(rest_setx)
    
    er_arr=(1/2)*np.mean((arr-rest_sety)**2)
    
    return (er_arr)    