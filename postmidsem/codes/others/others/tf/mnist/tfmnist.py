import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from mnist import MNIST
mndata = MNIST('./dataset')
traintup=mndata.load_training()


#print(traintup)
traindata=traintup[0]#60000 X 784
traindata=list(traindata)
traindata=np.array(traindata)
trainlabel=traintup[1] #vector of 60000
trainlabel=list(trainlabel)
trainlabel=np.array(trainlabel)

# Normalizing

# Convert string column to float

traindata=traindata.astype(float)

means= traindata.mean(axis=0)

stdevs=np.std(traindata,axis=0)

 
# standardize dataset
def standardize_dataset(traindata, means, stdevs):
	for row in traindata:
		for i in range(len(row)):

			row[i] = (row[i] - means[i])
			if stdevs[i]:
				row[i]/=stdevs[i]
standardize_dataset(traindata,means,stdevs)#this changes traindata
"""
traindata=traindata[:3000]
trainlabel=trainlabel[:3000]
"""

testtup=mndata.load_testing()

testdata=testtup[0]#60000 X 784

testdata=list(testdata)
testdata=np.array(testdata)
#print( "testdata", testdata.shape)
testlabel=testtup[1] #vector of 60000
testlabel=list(testlabel)
testlabel=np.array(testlabel)
print( "testlabel", testlabel.shape)
# Normalizing

# Convert string column to float

testdata=testdata.astype(float)

means= testdata.mean(axis=0)

stdevs=np.std(testdata,axis=0)

 
# standardize dataset

standardize_dataset(testdata,means,stdevs)#this changes testdata



newtraintup=(traindata,trainlabel)

print(trainlabel[0])
inputdim=784
outputdim=10

n_hid=500
batch_num=10
train_input_size=traindata.shape[0]
batch_size=train_input_size//batch_num
training_steps=20
newtrainlabel=np.zeros((train_input_size,outputdim))
for i in range(train_input_size):
	newtrainlabel[i][trainlabel[i]]=1
"""
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
mini=100
"""
x=tf.placeholder('float',[None,inputdim])
tar=tf.placeholder('int8')
par=tf.placeholder('int8')
w1=tf.Variable(tf.random_normal([inputdim, n_hid]))
b1=tf.Variable(tf.random_normal([n_hid,]))
w2=tf.Variable(tf.random_normal([ n_hid,outputdim]))
b2=tf.Variable(tf.random_normal([outputdim,]))
midout=tf.nn.relu(tf.add(tf.matmul(x,w1),b1))

prediction=tf.add(tf.matmul(midout,w2),b2)
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print(sess.run(prediction,feed_dict={x:traindata[:1000,:],tar:newtrainlabel[:1000,:]}))

cost=tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=tar)
cost=tf.reduce_mean(cost)
optmzr = tf.train.AdamOptimizer().minimize(cost)
    
#optmzr=tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost,var_list=[w1,b1,w2,b2])
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print(sess.run(cost,feed_dict={x:traindata[:1000,:],tar:newtrainlabel[:1000,:]}))
	for epoch in range(training_steps):
	            epoch_loss = 0
	            for i in range(batch_num):
	                epoch_x, epoch_y = traindata[i*batch_size:(i+1)*batch_size,:],newtrainlabel[i*batch_size:(i+1)*batch_size]
	                _, c = sess.run([optmzr, cost], feed_dict={x: epoch_x, tar: epoch_y})
	                epoch_loss += c

	            print('Epoch', epoch, 'completed out of',training_steps,'loss:',epoch_loss)

	correct = tf.equal(tf.cast(tf.argmax(prediction, axis=1),'int8'), par)

	accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
	print('Accuracy:',accuracy.eval({x:testdata, par:testlabel}))
