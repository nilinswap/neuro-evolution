
import theano
import theano.tensor as T
import numpy as np
import pylab as pl
def myrange(start,end,step):
    i=start
    while i+step < end:
        i+=step
        yield i
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



    
    pimadata=np.loadtxt("pima_dataset.csv", delimiter=',')
    
    np.random.shuffle(pimadata)
    
    nin =8;
    nout=2;
    pimadata=pimadata.astype(float)
    traindata=pimadata
    means= traindata.mean(axis=0)

    stdevs=np.std(traindata,axis=0)
    # standardize dataset
    standardize_dataset(traindata[:,:8],means,stdevs)
    rest_setx=pimadata[:538,:8]#tuple of two shared variable of array
    rest_sety=pimadata[:538,8:]
    test_setx=pimadata[538:,:8]
    test_sety=pimadata[538:,8:]
  
    return ((rest_setx,rest_sety),(test_setx,test_sety))
def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
       
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')
def give_datainshared():
    pimadata=np.loadtxt("pima_dataset.csv", delimiter=',')
    
    np.random.shuffle(pimadata)
    
    nin =8;
    nout=1;
    pimadata=pimadata.astype(float)
    traindata=pimadata
    means= traindata.mean(axis=0)

    stdevs=np.std(traindata,axis=0)
    # standardize dataset
    standardize_dataset(traindata[:,:8],means,stdevs)
    rest_setx=pimadata[:431,:8]#tuple of two shared variable of array
    rest_sety=pimadata[:431,8:]
    rest_sety=rest_sety.reshape((rest_sety.shape[0],))
    test_setx=pimadata[431:538,:8]
    test_sety=pimadata[431:538,8:]
    test_sety=test_sety.reshape((test_sety.shape[0],))
    print(rest_sety.reshape((rest_sety.shape[0],)))
    srest_setx,srest_sety=shared_dataset((rest_setx,rest_sety))
    stest_setx,stest_sety=shared_dataset((test_setx,test_sety))
    return ((srest_setx,srest_sety),(stest_setx,stest_sety))
def main():
    print(give_data()[1])
if __name__=="__main__":
    main()
