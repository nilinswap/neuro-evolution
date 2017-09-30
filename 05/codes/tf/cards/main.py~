from __future__ import print_function

__docformat__ = 'restructedtext en'

import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit
#from theano.ifelse import ifelse
import numpy as np
import tensorflow as tf
#import theano
#import theano.tensor as T
import tf_load_data

import tf_hiddenlayer
from tf_mlp import MLP

def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
             dataset='cards.data', n_par=10, n_hidden=100,freq_par=0.1):
	
	savo=None          #so that 'if' block sees the global savo
	restn=520
	testn=133
	nin=15
	nout=2
	rest_setx=tf.Variable(initial_value=np.zeros((restn,nin)),name='rest_setx',dtype=tf.float64)
	rest_sety=tf.Variable(initial_value=np.zeros((restn,)),name='rest_sety',dtype=tf.int32)
	test_setx=tf.Variable(initial_value=np.zeros((testn,nin)),name='rest_sety',dtype=tf.float64)
	test_sety=tf.Variable(initial_value=np.zeros((testn,)),name='test_sety',dtype=tf.int32)
	if not os.path.isfile('/home/robita/forgit/neuro-evolution/05/state/tf/cards/model.ckpt.meta'):
		lis=tf_load_data.load_data(dataset)
		
		rest_set=lis[1]#tuple of two shared variable of array
		test_set=lis[0]#tuple of shared variable of array

		millis=lis[2]#lis

		rxn=rest_setx.assign(rest_set[0])
		ryn=rest_sety.assign(rest_set[1])
		txn=test_setx.assign(test_set[0])
		tyn=test_sety.assign(test_set[1])
		var_lis=[rest_setx,rest_sety,test_setx,test_sety]
		nodelis=[rxn,ryn,txn,tyn]
		savo=tf.train.Saver(var_list=var_lis)
		with tf.Session() as sess:
			sess.run([i for i in nodelis])
			print("saving checkpoint")
			save_path = savo.save(sess, "/home/robita/forgit/neuro-evolution/05/state/tf/cards/model.ckpt")

	
	savo=tf.train.Saver(var_list=[rest_setx,rest_sety,test_setx,test_sety])
	par_size=int(rest_setx.shape[0])//n_par
	#print(rest_set[0].get_value().shape[0])
	#
	prmsdind=tf.placeholder('int32')
	
	
	valid_x_to_be=rest_setx[prmsdind*par_size:(prmsdind+1)*par_size,:]
	valid_y_to_be=rest_sety[prmsdind*par_size:(prmsdind+1)*par_size]
	train_x_to_be=tf.concat((rest_setx[:(prmsdind)*par_size,:],rest_setx[(prmsdind+1)*par_size:,:]),axis=0)
	train_y_to_be=tf.concat((rest_sety[:(prmsdind)*par_size],rest_sety[(prmsdind+1)*par_size:]),axis=0)
	# allocate symbolic variables for the data
		#index = T.lscalar()  # index to a [mini]batch
	x = tf.placeholder(dtype=tf.float64,name='x',shape=[None,nin]) 
	y = tf.placeholder(dtype=tf.int32,name='y')  

	rng = np.random.RandomState(1234)

	##yen_hidden=T.lscalar()
	n_hid=5
	classifier= MLP(
		    rng=rng,
		    input=x,
		    n_in=nin,
		    n_hidden=n_hid,
		    n_out=nout		#this was important, I tried taking 1 gave fatal results, decided to fix this later.
		)

	cost = tf.add(
	    classifier.negative_log_likelihood(y)
	    ,L1_reg * classifier.L1
	    ,L2_reg * classifier.L2_sqr
	)

	

	optmzr = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost,var_list=[valid_x_to_be,valid_y_to_be,train_x_to_be,train_y_to_be])

	with tf.Session() as sess:
		savo.restore(sess, "/home/robita/forgit/neuro-evolution/05/state/tf/cards/model.ckpt")
		print("------------------------------------------------------------------")
		print(sess.run([valid_x_to_be,valid_y_to_be,train_x_to_be,train_y_to_be],feed_dict={prmsdind:0}))
	

		#cool thing starts from here ->
	    ######################
	    # BUILD ACTUAL MODEL #
	    ######################

		print('...building the model')

		
		for epoch in range(n_epochs):
			for ind in range(n_par):
				sess.run([optmzr,cost],feed_dict={prmsdind:ind,x:train_x_to_be,y:train_y_to_be})
			if epoch%10==0:
				print(sess.run([classifier.errors(y)],feed_dict={prmsdind:ind,x:valid_x_to_be,y:valid_y_to_be}))
		#setting condition to increase frequency of validation with epochs done as first thing inside 'i in n_epochs' loop
	"""
		epo=T.lscalar()

		epostep=theano.shared(100)
		eqh=ifelse(T.eq(epo%epostep,0),1,0)
		ie=ifelse(T.lt(1,epostep),eqh,1)
		booly=theano.function([epo],ie)

		#here it ends

		# end-snippet-5

		###############
		# TRAIN MODEL #
		###############
		minavg=np.inf
		lasttolastavg=10002
		lastavg=10001
		presentavg=10000
		flag=0

		for i in range(n_epochs):
			if booly(i):
				epostep.set_value(int(epostep.get_value()*freq_par))
				
				avrg=0
			for ind in range(n_par):
				trainerr=train_model(ind)
				if booly(i):
					validerr=validate_model(ind)
					avrg=(avrg*(ind)+validerr)/(ind+1)
			if booly(i):
		 		
		 		lasttolastavg=lastavg
		 		lastavg=presentavg
		 		presentavg=avrg
		 		print(presentavg)
		 		if not ((( lastavg- presentavg) > 0.001) or ((lasttolastavg - lastavg)>0.001)):
		 			flag=1
			if flag:
				break

		if not flag:
			print("increase epochs maybe")
		print("achieved min err as", presentavg, "for",n_hid,"hidden nodes")
		
		if avrg<minminavg:
			minclassifier=classifier
			minminavg=avrg
	print("minavg was ",minminavg , "and for ", minclassifier.n_hidden, "nodes")
	print("test error is ",test_model(0))
	"""
	"""minminavg=np.inf
	for n_hid in range(0,n_hidden,10):
			# construct the MLP class
		classifier= MLP(
		    rng=rng,
		    input=x,
		    n_in=nin,
		    n_hidden=n_hid,
		    n_out=2			#this was important, I tried taking 1 gave fatal results, decided to fix this later.
		)
		
		# start-snippet-4
		# the cost we minimize during training is the negative log likelihood of
		# the model plus the regularization terms (L1 and L2); cost is expressed
		# here symbolically
		cost = tf.add(
		    classifier.negative_log_likelihood(y)
		    ,L1_reg * classifier.L1
		    ,L2_reg * classifier.L2_sqr
		)
		# end-snippet-4

		# compiling a Theano function that computes the mistakes that are made
		# by the model on a minibatch
		
		test_model = theano.function(
		    inputs=[prmsdind],#this was not really required for testset
		    outputs=classifier.errors(y),
		    givens={
		    	x:test_x_to_be,
		    	y:test_y_to_be
		    },
		    on_unused_input='ignore'
		)

		validate_model = theano.function(
		    inputs=[prmsdind],
		    outputs=classifier.errors(y),
		    givens={
		        x: valid_x_to_be,
		        y: valid_y_to_be
		    }
		)

		# start-snippet-5
		# compute the gradient of cost with respect to theta (sorted in params)
		# the resulting gradients will be stored in a list gparams
		gparams = [T.grad(cost, param) for param in classifier.params]

		# specify how to update the parameters of the model as a list of
		# (variable, update expression) pairs

		# given two lists of the same length, A = [a1, a2, a3, a4] and
		# B = [b1, b2, b3, b4], zip generates a list C of same size, where each
		# element is a pair formed from the two lists :
		#    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
		updates = [
		    (param, param - learning_rate * gparam)
		    for param, gparam in zip(classifier.params, gparams)
		]

		# compiling a Theano function `train_model` that returns the cost, but
		# in the same time updates the parameter of the model based on the rules
		# defined in `updates`
		train_model = theano.function(
		    inputs=[prmsdind],
		    outputs=cost,
		    updates=updates,
		    givens={
		        x: train_x_to_be,
		        y: train_y_to_be 
		        }
		)

		#setting condition to increase frequency of validation with epochs done as first thing inside 'i in n_epochs' loop
		epo=T.lscalar()

		epostep=theano.shared(100)
		eqh=ifelse(T.eq(epo%epostep,0),1,0)
		ie=ifelse(T.lt(1,epostep),eqh,1)
		booly=theano.function([epo],ie)

		#here it ends

		# end-snippet-5

		###############
		# TRAIN MODEL #
		###############
		minavg=np.inf
		lasttolastavg=10002
		lastavg=10001
		presentavg=10000
		flag=0

		for i in range(n_epochs):
			if booly(i):
				epostep.set_value(int(epostep.get_value()*freq_par))
				
				avrg=0
			for ind in range(n_par):
				trainerr=train_model(ind)
				if booly(i):
					validerr=validate_model(ind)
					avrg=(avrg*(ind)+validerr)/(ind+1)
			if booly(i):
		 		
		 		lasttolastavg=lastavg
		 		lastavg=presentavg
		 		presentavg=avrg
		 		print(presentavg)
		 		if not ((( lastavg- presentavg) > 0.001) or ((lasttolastavg - lastavg)>0.001)):
		 			flag=1
			if flag:
				break

		if not flag:
			print("increase epochs maybe")
		print("achieved min err as", presentavg, "for",n_hid,"hidden nodes")
		
		if avrg<minminavg:
			minclassifier=classifier
			minminavg=avrg
	print("minavg was ",minminavg , "and for ", minclassifier.n_hidden, "nodes")
	print("test error is ",test_model(0))"""
def main():
 	test_mlp(n_hidden=200)
if __name__ == '__main__':
    main()

