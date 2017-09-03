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

def standardize_dataset(traindata, means, stdevs):
    for row in traindata:
        for i in range(len(row)):

            row[i] = (row[i] - means[i])
            if stdevs[i]:
                row[i]/=stdevs[i]
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

def load_data():
	convert_iris()
	irisdata=np.loadtxt("/home/swapnil/forgit/neuro-evolution/05/dataset/iris/newiris.data", delimiter=',')
	order=np.arange(np.shape(irisdata)[0])
	np.random.shuffle(order)
	irisdata = irisdata[order,:]
	irisdata=irisdata.astype(float)
	traindata=irisdata
	means= traindata.mean(axis=0)

	stdevs=np.std(traindata,axis=0)

	 
	# standardize dataset

	standardize_dataset(traindata[:,:4],means,stdevs)
	rest_setx=irisdata[:120,:4]#tuple of two shared variable of array
	rest_sety=irisdata[:120,:4]
	test_setx=irisdata[120:,4]
	test_sety=irisdata[120:,4]

	test_setx, test_sety = shared_dataset((test_setx,test_sety))
	rest_setx, rest_sety = shared_dataset((rest_setx,rest_sety))
	print(type(test_setx))
	return ((rest_setx,rest_sety),(test_setx,test_sety))

