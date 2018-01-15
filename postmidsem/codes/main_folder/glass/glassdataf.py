import random
import numpy as np
#import pylab as pl

def standardize_dataset(traindata, means, stdevs):
    for row in traindata:
        for i in range(len(row)):

            row[i] = (row[i] - means[i])
            if stdevs[i]:
                row[i]/=stdevs[i]
rng=random

glassdata=np.loadtxt("glass.csv", delimiter=',')

#rng.shuffle(glassdata) this was a big big error
numlis = np.arange(glassdata.shape[0])
rng.shuffle(numlis)
glassdata = glassdata[ numlis ]


glassdata=glassdata.astype(float)
traindata=glassdata
means= traindata.mean(axis=0)

stdevs=np.std(traindata,axis=0)
# standardize dataset
standardize_dataset(traindata[:,:9],means,stdevs)

def get_dimension():
    in_dem = 9
    out_dem = 7
    return (in_dem, out_dem)

def myrange(start,end,step):
    i=start
    while i+step < end:
        i+=step
        yield i
#print(traindata)

def give_data():
    #1. make iris.data in usable form
    #2. make input set and output set out of it
    #3. make setpool out of the dataset
    #4. make pcn and train it
    #5. test on validation and testing set    
    rest_setx=glassdata[:160,:9]#tuple of two shared variable of array
    rest_sety=glassdata[:160,9:]
    test_setx=glassdata[160:,:9]
    test_sety=glassdata[160:,9:]
    #print(glassdata.shape)
    #print(rest_setx.shape,test_setx.shape)
    return ((rest_setx,rest_sety),(test_setx,test_sety))

def main():
    print(give_data()[1])

if __name__ == "__main__":
    main()
