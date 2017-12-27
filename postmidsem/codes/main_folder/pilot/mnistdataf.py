import random
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from mnist import MNIST





def standardize_dataset(traindata, means, stdevs):
	for row in traindata:
		for i in range(len(row)):

			row[i] = (row[i] - means[i])
			if stdevs[i]:
				row[i] /= stdevs[i]


rng = random

#pimadata = np.loadtxt("pima_dataset.csv", delimiter=',')
mndata = MNIST('./mnist_dataset')

##training preparation starts
traintup=mndata.load_training()
traindata=traintup[0]#60000 X 784
traindata=list(traindata)
traindata=np.array(traindata)



trainlabel=traintup[1] #vector of 60000
trainlabel=list(trainlabel)
trainlabel=np.array(trainlabel)
trainlabel = trainlabel.reshape( (trainlabel.shape[0], 1) )

#traindata = pimadata
clumped_training = np.concatenate( (traindata, trainlabel), axis = 1)
clumped_training = clumped_training.astype(float)
rng.shuffle(clumped_training)

traindata_new = clumped_training[:,:-1]
trainlabel_new = clumped_training[:,-1:]



means = traindata_new.mean(axis=0)

stdevs = np.std(traindata_new, axis=0)
# standardize dataset
standardize_dataset(traindata_new, means, stdevs)

##training data prepared

##testing data preparation starts

testtup=mndata.load_testing()
testdata=testtup[0]#10000 X 784
testdata=list(testdata)
testdata=np.array(testdata)



testlabel=testtup[1] #vector of 10000
testlabel=list(testlabel)
testlabel=np.array(testlabel)
testlabel = testlabel.reshape( (testlabel.shape[0], 1) )

#testdata = pimadata
clumped_testing = np.concatenate( (testdata, testlabel), axis = 1)
clumped_testing = clumped_testing.astype(float)
rng.shuffle(clumped_testing)

testdata_new = clumped_testing[:,:-1]
testlabel_new = clumped_testing[:,-1:]



means = testdata_new.mean(axis=0)

stdevs = np.std(testdata_new, axis=0)
# standardize dataset
standardize_dataset(testdata_new, means, stdevs)

##testing data prepared







def myrange(start, end, step):
	i = start
	while i + step < end:
		i += step
		yield i


# print(traindata)
def give_data():
	# 1. make iris.data in usable form
	# 2. make input set and output set out of it
	# 3. make setpool out of the dataset
	# 4. make pcn and train it
	# 5. test on validation and testing set


	return ((traindata_new, trainlabel_new), (testdata_new, testlabel_new))


def main():
	print(give_data()[1][0][4].tolist() )


if __name__ == "__main__":
	main()
